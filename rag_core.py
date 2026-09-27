#!/usr/bin/env python3

import os
import sys
import warnings
import pandas as pd
import pickle
import logging
import json
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
## Removed unused imports: langdetect, deep_translator, wordnet
import streamlit as st

# Import all the classes and functions from your rag.py
from rag import (
    CONFIG, HallucinationDetector, CachedHuggingFaceEmbeddings, 
    CachedRetriever, DeduplicatingRetriever, DocumentChangeHandler,
    start_document_watcher, generate_content_hash, deduplicate_documents,
    process_single_document, process_documents_parallel, process_document_file,
    create_vector_db, initialize_bm25_retriever, load_bm25_retriever,
    DB_FAISS_PATH, BM25_RETRIEVER_PATH, CACHE_DIR, DOCUMENT_HASH_FILE,
    embeddings_cache, query_cache, document_hashes, logger, build_rag_prompt
)

from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from Ingestion.ingest import extract_text_and_metadata_from_pdf_document, extract_text_and_metadata_from_docx_document
from PIL import Image
import pytesseract

# Suppress warnings
warnings.filterwarnings("ignore")

class RAGSystem:
    """
    Comprehensive RAG System wrapper that provides all functionality from rag.py
    in a streamlit-friendly interface
    """
    
    def __init__(self):
        self.embedding_model = None
        self.hallucination_detector = None
        self.qa_chain = None
        self.retriever = None
        self.observer = None
        self.initialized = False
        self.documents = []
        self.processing_stats = {
            'total_documents': 0,
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'hallucinations_detected': 0
        }
        
        logger.info("RAG System initialized")
        
    def initialize_system(self, enable_monitoring=True):
        """Initialize the complete RAG system"""
        try:
            logger.info("Loading embedding model...")
            self.embedding_model = self._load_embedding_model()
            
            logger.info("Loading hallucination detector...")
            self.hallucination_detector = HallucinationDetector()
            
            # Start document monitoring if requested
            if enable_monitoring and not self.observer:
                self.observer = start_document_watcher(".")
                
            logger.info("RAG System fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            return False
    
    def _load_embedding_model(self):
        """Load cached embedding model on CPU"""
        return CachedHuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1"
        )
    
    def add_documents(self, uploaded_files, use_parallel=True):
        """Process uploaded documents and add to knowledge base, and persist to DB"""
        from database import get_db_manager, Document as DB_Document
        db_manager = get_db_manager()
        if db_manager is None:
            logger.error("db_manager is None. Database not initialized.")
        stats = {
            'new_documents': 0,
            'total_documents': 0,
            'errors': [],
            'processing_time': 0
        }
        start_time = time.time()
        try:
            if not self.embedding_model:
                if not self.initialize_system():
                    stats['errors'].append("Failed to initialize system")
                    return stats
            # Save uploaded files temporarily
            temp_files = []
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append((temp_path, uploaded_file.type))
            # Process documents
            new_documents = []
            if use_parallel and len(temp_files) > 1:
                logger.info("Using parallel document processing")
                file_infos = [(path, 'pdf' if 'pdf' in file_type else 'docx') 
                             for path, file_type in temp_files]
                new_documents = process_documents_parallel(file_infos)
            else:
                logger.info("Using sequential document processing")
                for temp_path, file_type in temp_files:
                    file_format = 'pdf' if 'pdf' in file_type else 'docx'
                    doc_batch = []
                    process_document_file(temp_path, file_format, doc_batch)
                    new_documents.extend(doc_batch)
            # Deduplicate documents
            new_documents = deduplicate_documents(new_documents)
            self.documents.extend(new_documents)
            self.documents = deduplicate_documents(self.documents)
            stats['new_documents'] = len(new_documents)
            stats['total_documents'] = len(self.documents)
            logger.info(f"[DEBUG] Number of document chunks to index: {len(self.documents)}")
            for i, doc in enumerate(self.documents[:3]):
                logger.info(f"[DEBUG] Indexed chunk {i+1}: page_number={doc.metadata.get('page_number')}, content[:200]={repr(doc.page_content[:200])}")
            # Persist documents to DB
            if db_manager is not None:
                session = db_manager.get_session()
                for doc in new_documents:
                    try:
                        db_doc = db_manager.create_document(
                            filename=doc.metadata.get('filename', 'Unknown'),
                            file_path=doc.metadata.get('filename', 'Unknown'),
                            file_hash=doc.metadata.get('content_hash', ''),
                            file_type=doc.metadata.get('type', 'text'),
                            owner_id=1,  # Default to user id 1; update as needed
                            file_size=len(doc.page_content)
                        )
                    except Exception as e:
                        logger.error(f"Error saving document to DB: {e}")
                session.close()
            else:
                logger.error("db_manager is None. Skipping DB persistence.")
            # Create/update vector store and retrievers
            if new_documents:
                self._update_knowledge_base()
                self.initialized = True
            stats['processing_time'] = time.time() - start_time
            self.processing_stats['total_documents'] = len(self.documents)
            logger.info(f"Added {stats['new_documents']} new documents")
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            stats['errors'].append(str(e))
        # Clean up temp files AFTER all processing
        for temp_path, _ in temp_files:
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return stats
    
    def _update_knowledge_base(self):
        """Update vector store and retrievers with current documents"""
        try:
            # Create vector store
            create_vector_db(self.documents, self.embedding_model)
            
            # Initialize BM25 retriever
            bm25_retriever = initialize_bm25_retriever(self.documents)
            
            # Load FAISS vector store
            db = FAISS.load_local(DB_FAISS_PATH, self.embedding_model, 
                                allow_dangerous_deserialization=True)
            
            # Setup retrievers
            faiss_retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": CONFIG['RETRIEVAL_TOP_K'],
                    "fetch_k": CONFIG['RETRIEVAL_TOP_K'] * 4,
                    "lambda_mult": 0.3
                }
            )
            
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], 
                weights=[0.5, 0.5]
            )
            
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.embedding_model,
                similarity_threshold=CONFIG['SIMILARITY_THRESHOLD']
            )
            
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=embeddings_filter,
                base_retriever=ensemble_retriever
            )
            
            cached_retriever = CachedRetriever(compression_retriever)
            self.retriever = DeduplicatingRetriever(cached_retriever)
            
            # Setup QA chain
            self._setup_qa_chain()
            
            logger.info("Knowledge base updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base: {e}")
            raise
    
    def _setup_qa_chain(self):
        """Setup the QA chain with language model"""
        try:
            use_cuda = CONFIG['DEVICE'] == 'cuda' and torch.cuda.is_available()
            from rag import build_generation_pipeline_with_retries, build_rag_prompt
            from langchain.chains import LLMChain
            from langchain_huggingface import HuggingFacePipeline

            class StringOnlyHFPipeline(HuggingFacePipeline):
                def __call__(self, prompt, **kwargs):
                    if isinstance(prompt, dict):
                        # Format the prompt using the template if needed
                        raise ValueError("StringOnlyHFPipeline received a dict. It should only receive a string.")
                    return super().__call__(prompt, **kwargs)

            pipe = build_generation_pipeline_with_retries()
            prompt = build_rag_prompt()
            llm = StringOnlyHFPipeline(pipeline=pipe)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=HuggingFacePipeline(pipeline=pipe),
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
                return_source_documents=True,
            )
            logger.info(f"QA chain setup complete with {CONFIG['MODEL_ID']}")
        except Exception as e:
            logger.error(f"Error setting up QA chain: {e}")
            raise
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Process a query and return comprehensive results, and log to DB"""
        from database import db_manager
        if not self.initialized or not self.qa_chain:
            return {
                'answer': 'System not initialized. Please upload documents first.',
                'source_documents': [],
                'hallucination_info': {},
                'processing_time': 0
            }
        start_time = time.time()
        try:
            logger.info(f"[DEBUG] Query string: {repr(query)}")
            # Prepare input for the QA chain
            # The QA chain expects both 'context' and 'question' as input variables.
            # The retriever will fill in 'context'.
            input_dict = {"query": query, "context": ""}
            try:
                result = self.qa_chain.invoke(input_dict)
            except Exception:
                result = self.qa_chain(input_dict)
            response = (result.get('result') or 
                       result.get('answer') or 
                       result.get('output_text') or 
                       str(result))
            source_docs = result.get('source_documents', []) or []
            logger.info(f"[DEBUG] Number of source documents returned for query: {len(source_docs)}")
            for i, doc in enumerate(source_docs[:3]):
                logger.info(f"[DEBUG] Retrieved chunk {i+1}: page_number={doc.metadata.get('page_number')}, content[:200]={repr(doc.page_content[:200])}")
            # Detect hallucinations
            is_hallucination, confidence, similar_sentence = \
                self.hallucination_detector.detect(response, source_docs)
            # Update stats
            self.processing_stats['total_queries'] += 1
            if is_hallucination:
                self.processing_stats['hallucinations_detected'] += 1
            # Get cache stats
            try:
                cache_stats = self.retriever.base_retriever.get_cache_stats()
                self.processing_stats.update(cache_stats)
            except Exception:
                pass
            processing_time = time.time() - start_time
            # Log query to DB
            try:
                db_manager.log_query(
                    user_id=1,  # Default to user id 1; update as needed
                    query_text=query,
                    response_text=response,
                    confidence_score=int((1 - confidence) * 100),
                    is_hallucination=is_hallucination,
                    processing_time=int(processing_time * 1000),
                    source_documents=[doc.metadata.get('id', 0) for doc in source_docs]
                )
            except Exception as e:
                logger.error(f"Error logging query to DB: {e}")
            return {
                'answer': response,
                'source_documents': source_docs,
                'hallucination_info': {
                    'is_hallucination': is_hallucination,
                    'confidence': confidence,
                    'similar_sentence': similar_sentence
                },
                'processing_time': processing_time,
                'cache_info': self._get_cache_info()
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f'Error processing query: {str(e)}',
                'source_documents': [],
                'hallucination_info': {},
                'processing_time': time.time() - start_time
            }
    
    def _get_cache_info(self):
        """Get cache statistics"""
        try:
            cache_info = {}
            if self.embedding_model:
                cache_info['embeddings'] = self.embedding_model.get_cache_stats()
            
            try:
                cache_info['embeddings_size'] = embeddings_cache.volume()
                cache_info['query_size'] = query_cache.volume()
            except Exception:
                pass
                
            return cache_info
        except Exception:
            return {}
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        stats = self.processing_stats.copy()
        stats.update({
            'initialized': self.initialized,
            'documents_loaded': len(self.documents),
            'cache_info': self._get_cache_info(),
            'config': CONFIG
        })
        return stats
    
    def export_results(self, query_history: List[Dict]) -> str:
        """Export query results to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_results_{timestamp}.csv"
            
            # Prepare data for export
            export_data = []
            for i, item in enumerate(query_history):
                export_data.append({
                    'Query_No': i + 1,
                    'Query': item['query'],
                    'Response': item['response'],
                    'Timestamp': item['time'],
                    'Is_Hallucination': item.get('is_hallucination', False),
                    'Hallucination_Confidence': item.get('hallucination_confidence', 0.0),
                    'Processing_Time': item.get('processing_time', 0.0)
                })
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"Results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""
    
    def clear_cache(self):
        """Clear all caches"""
        try:
            embeddings_cache.clear()
            query_cache.clear()
            logger.info("Caches cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def shutdown(self):
        """Cleanup resources"""
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            embeddings_cache.close()
            query_cache.close()
            
            logger.info("RAG System shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
# --- OCR and Unstructured Extraction Utilities (DocTR only) ---
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx

def extract_text_from_image_doctr(image_path):
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_images(image_path)
    result = model(doc)
    return result.export()['value']

def extract_text_from_pdf_with_ocr(pdf_path):
    elements = partition_pdf(pdf_path)
    text = ""
    for el in elements:
        if hasattr(el, 'text') and el.text:
            text += el.text + "\n"
        elif hasattr(el, 'image') and el.image:
            text += extract_text_from_image_doctr(el.image) + "\n"
    return text

def extract_text_from_docx_with_ocr(docx_path):
    elements = partition_docx(docx_path)
    text = ""
    for el in elements:
        if hasattr(el, 'text') and el.text:
            text += el.text + "\n"
        elif hasattr(el, 'image') and el.image:
            text += extract_text_from_image_doctr(el.image) + "\n"
    return text

# --- Integrate into Document Processing ---
# --- PII Redaction Utilities ---
import re
import spacy
nlp = spacy.load("en_core_web_sm")

PII_PATTERNS = [
    (r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<REDACTED_EMAIL>"), # Email
    (r"\b(?:\d[ -]*?){13,16}\b", "<REDACTED_CREDIT_CARD>"), # Credit Card
]

PII_ENTITY_LABELS = {"PERSON": "<REDACTED_NAME>", "GPE": "<REDACTED_LOCATION>", "ORG": "<REDACTED_ORG>"}

def redact_pii(text):
    # Regex-based redaction
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    # NER-based redaction
    doc = nlp(text)
    redacted = text
    for ent in doc.ents:
        if ent.label_ in PII_ENTITY_LABELS:
            redacted = redacted.replace(ent.text, PII_ENTITY_LABELS[ent.label_])
    return redacted

def process_document_file(file_path, file_type, documents):
    try:
        if file_type == 'pdf':
            from Ingestion.ingest import extract_text_and_metadata_from_pdf_document
            df = extract_text_and_metadata_from_pdf_document(file_path)
            for _, row in df.iterrows():
                text = row['Text']
                if text and text.strip():
                    text = redact_pii(text)
                    document = Document(
                        page_content=text,
                        metadata={
                            'id': row.get('Parent_Id', os.path.basename(file_path)),
                            'type': row.get('Element_Type', 'text'),
                            'filename': row.get('Filename', os.path.basename(file_path)),
                            'page_number': row.get('Page_Number', 1),
                            'content_hash': generate_content_hash(text)
                        }
                    )
                    documents.append(document)
        elif file_type == 'docx':
            from Ingestion.ingest import extract_text_and_metadata_from_docx_document
            df = extract_text_and_metadata_from_docx_document(file_path)
            for _, row in df.iterrows():
                text = row['Text']
                if text and text.strip():
                    text = redact_pii(text)
                    document = Document(
                        page_content=text,
                        metadata={
                            'id': row.get('Parent_Id', os.path.basename(file_path)),
                            'type': row.get('Element_Type', 'text'),
                            'filename': row.get('Filename', os.path.basename(file_path)),
                            'page_number': row.get('Page_Number', 1),
                            'content_hash': generate_content_hash(text)
                        }
                    )
                    documents.append(document)
        elif file_type in ['jpg', 'jpeg', 'png', 'tiff']:
            text = extract_text_from_image_doctr(file_path)
            if text:
                text = redact_pii(text)
                document = Document(
                    page_content=text,
                    metadata={
                        'id': os.path.basename(file_path),
                        'type': 'image',
                        'filename': os.path.basename(file_path),
                        'page_number': 1,
                        'content_hash': generate_content_hash(text)
                    }
                )
                documents.append(document)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def strict_answer_generation(context, question, answer):
    # Only return answer if context contains relevant info
    if not context or (answer.strip().lower() in ["i don't know", "not found", "no answer"]):
        return "I don't know. Please provide more information."
    # Check if answer is directly supported by context
    context_lower = context.lower()
    answer_lower = answer.lower()
    if any(word in context_lower for word in answer_lower.split()):
        return answer
    return "I don't know. Please provide more information."
