#!/usr/bin/env python3
import argparse
import os
import sys
import warnings
import pandas as pd
import pickle
from tqdm.auto import tqdm
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_huggingface import HuggingFacePipeline
from Ingestion.ingest import extract_text_and_metadata_from_pdf_document, extract_text_and_metadata_from_docx_document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
import torch
import hashlib
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from diskcache import Cache
from pydantic import Field
import time
from filelock import FileLock
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Tuple, Optional
from cryptography.fernet import Fernet
import base64
import secrets
import bcrypt
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from database import db_manager, User
import spacy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration parameters
# Embedding and hallucination detection always on CPU
EMBEDDING_DEVICE = 'cpu'
HALLUCINATION_DEVICE = 'cpu'
# LLM on GPU if available
LLM_DEVICE = 0 if torch.cuda.is_available() else -1

CONFIG = {
    'BATCH_SIZE': 32,
    'SIMILARITY_THRESHOLD': 0.5,
    'MAX_NEW_TOKENS': 200,
    'CACHE_EXPIRATION': 3600,
    'EMBEDDINGS_CACHE_SIZE_LIMIT': 2 * 1024 * 1024 * 1024,
    'QUERY_CACHE_SIZE_LIMIT': 1 * 1024 * 1024 * 1024,
    'CACHE_EVICTION_POLICY': 'least-recently-used',
    'MODEL_ID': 'microsoft/Phi-3-mini-128k-instruct',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'HALLUCINATION_THRESHOLD': 0.35,  # Similarity threshold for hallucination detection
    'MAX_WORKERS': max(1, cpu_count() - 2),  # Leave 2 cores free
    'CHUNK_SIZE': 4,  # Documents per worker in parallel processing (if using process pool)
    'RETRIEVAL_TOP_K': 6,
    'MIN_HALLUCINATION_CONFIDENCE': 0,  # Minimum confidence to flag as hallucination
    'TOP_P': 0.85,
    'REPETITION_PENALTY': 1.1,
    'QUANTIZATION': 'int4',
    'EMBEDDING_DEVICE': EMBEDDING_DEVICE,
    'HALLUCINATION_DEVICE': HALLUCINATION_DEVICE,
    'LLM_DEVICE': LLM_DEVICE,
}

# Cache configuration
CACHE_DIR = ".cache"
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_DIR, "embeddings")
QUERY_RESULTS_CACHE_DIR = os.path.join(CACHE_DIR, "query_results")
DOCUMENT_HASH_FILE = os.path.join(CACHE_DIR, "document_hashes.json")
DB_FAISS_PATH = 'vectorstore/db_faiss'
BM25_RETRIEVER_PATH = 'bm25_retriever.pkl'
HALLUCINATION_MODEL_PATH = 'hallucination_model'
HF_CACHE_DIR = os.path.join(CACHE_DIR, "hf")

# Create cache directories
os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)
os.makedirs(QUERY_RESULTS_CACHE_DIR, exist_ok=True)
os.makedirs(HALLUCINATION_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)

# Initialize caches
embeddings_cache = Cache(
    EMBEDDINGS_CACHE_DIR,
    size_limit=CONFIG['EMBEDDINGS_CACHE_SIZE_LIMIT'],
    eviction_policy=CONFIG['CACHE_EVICTION_POLICY']
)
query_cache = Cache(
    QUERY_RESULTS_CACHE_DIR,
    size_limit=CONFIG['QUERY_CACHE_SIZE_LIMIT'],
    eviction_policy=CONFIG['CACHE_EVICTION_POLICY']
)

# Document hashes for change detection
document_hashes = {}
if os.path.exists(DOCUMENT_HASH_FILE):
    try:
        with open(DOCUMENT_HASH_FILE, 'r') as f:
            document_hashes = json.load(f)
    except Exception as e:
        logger.warning(f"Could not read document hashes file: {e}")
        document_hashes = {}

nlp = spacy.load('en_core_web_sm')

def advanced_sentence_split(text):
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

def keyword_overlap(sent1, sent2):
    set1 = set(sent1.lower().split())
    set2 = set(sent2.lower().split())
    return len(set1 & set2) / max(1, len(set1 | set2))

def named_entity_match(sent1, sent2):
    ents1 = set([ent.text for ent in nlp(sent1).ents])
    ents2 = set([ent.text for ent in nlp(sent2).ents])
    return len(ents1 & ents2) / max(1, len(ents1 | ents2))

class HallucinationDetector:
    """Detects potential hallucinations in model responses"""
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):

        """Load the sentence transformer model for hallucination detection"""
        try:
            model_name = "all-mpnet-base-v2"  # high-quality sentence embedding model
            self.model = SentenceTransformer(model_name, cache_folder=HALLUCINATION_MODEL_PATH)
            logger.info(f"Loaded hallucination detection model (all-mpnet-base-v2) on {CONFIG['HALLUCINATION_DEVICE']}")
        except Exception as e:
            logger.error(f"Error loading hallucination detection model: {e}")
            raise
    
    def detect(self, response: str, source_docs: List[Document]) -> Tuple[bool, float, Optional[str]]:
        """
        Detect potential hallucinations in the response compared to source documents.
        Returns:
            (is_hallucination, confidence_score, most_similar_sentence)
        """
        if not self.model or not source_docs:
            return False, 0.0, None
        try:
            # Advanced sentence splitting
            response_sentences = advanced_sentence_split(response)
            source_text = " ".join([doc.page_content for doc in source_docs])
            source_sentences = advanced_sentence_split(source_text)
            if not response_sentences or not source_sentences:
                return False, 0.0, None
            # Context windowing: only compare to top 10 most similar source sentences
            response_embeddings = self.model.encode(response_sentences, convert_to_tensor=True)
            source_embeddings = self.model.encode(source_sentences, convert_to_tensor=True)
            cos_scores = util.cos_sim(response_embeddings, source_embeddings)
            # For each response sentence, find top 10 most similar source sentences
            multi_metrics = []
            for i, resp_sent in enumerate(response_sentences):
                row = cos_scores[i]
                top_indices = row.cpu().numpy().argsort()[-10:][::-1]
                best_metric = 0
                for idx in top_indices:
                    src_sent = source_sentences[idx]
                    sim = float(row[idx].cpu().numpy())
                    overlap = keyword_overlap(resp_sent, src_sent)
                    entity_match = named_entity_match(resp_sent, src_sent)
                    # Weighted sum of metrics
                    metric = 0.6 * sim + 0.2 * overlap + 0.2 * entity_match
                    best_metric = max(best_metric, metric)
                multi_metrics.append(best_metric)
            avg_metric = sum(multi_metrics) / len(multi_metrics)
            is_hallucination = avg_metric < CONFIG['HALLUCINATION_THRESHOLD']
            confidence = 1.0 - avg_metric
            min_score_idx = int(min(range(len(multi_metrics)), key=lambda i: multi_metrics[i]))
            most_similar_idx = int(cos_scores[min_score_idx].cpu().numpy().argmax())
            most_similar_sentence = source_sentences[most_similar_idx] if source_sentences else None
            flagged = is_hallucination and confidence > CONFIG['MIN_HALLUCINATION_CONFIDENCE']
            return flagged, confidence, most_similar_sentence
        except Exception as e:
            logger.error(f"Error in hallucination detection: {e}")
            return False, 0.0, None
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting (replaceable with nltk/spacy for production)"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[:200]  # cap to avoid huge compute

def torch_max_per_row(tensor):
    """Return max value per row for a 2D torch tensor, robust to CPU/GPU tensors"""
    try:
        import torch as _torch
        max_vals, _ = _torch.max(tensor, dim=1)
        return max_vals
    except Exception:
        # Fallback convert to numpy
        arr = tensor.cpu().numpy()
        return np.max(arr, axis=1)

def generate_content_hash(text: str) -> str:
    """Generate consistent SHA-256 hash for text content"""
    return hashlib.sha256(text.strip().encode('utf-8')).hexdigest()

def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Remove duplicate documents while preserving order"""
    seen_hashes = set()
    unique_docs = []
    for doc in documents:
        content_hash = generate_content_hash(doc.page_content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    logger.info(f"Deduplicated {len(documents)} -> {len(unique_docs)} documents")
    return unique_docs

class DeduplicatingRetriever(BaseRetriever):
    """Wrapper retriever that removes duplicate results"""
    base_retriever: BaseRetriever = Field(...)

    def __init__(self, base_retriever):
        super().__init__(base_retriever=base_retriever)

    def get_relevant_documents(self, query):
        results = self.base_retriever.get_relevant_documents(query)
        return deduplicate_documents(results)

    def _get_relevant_documents(self, query):
        return self.get_relevant_documents(query)

class DocumentChangeHandler(FileSystemEventHandler):
    """Handler for document changes to invalidate cache"""
    def __init__(self, dir_path):
        self.dir_path = dir_path
    
    def on_modified(self, event):
        self._handle_event(event, 'modified')
    
    def on_created(self, event):
        self._handle_event(event, 'created')
    
    def on_deleted(self, event):
        self._handle_event(event, 'deleted')
    
    def _handle_event(self, event, event_type):
        if not event.is_directory and (event.src_path.endswith('.pdf') or event.src_path.endswith('.docx')):
            logger.info(f"Detected {event_type} document: {event.src_path}")
            self.invalidate_cache_for_document(event.src_path)
    
    def invalidate_cache_for_document(self, file_path):
        """Invalidate cache entries for changed document"""
        file_hash = self._compute_file_hash(file_path)
        relative_path = os.path.relpath(file_path, self.dir_path)
        
        if relative_path in document_hashes and document_hashes[relative_path] == file_hash:
            logger.debug(f"No change in {relative_path} hash; skipping invalidation")
            return
        
        document_hashes[relative_path] = file_hash
        self._save_document_hashes()
        # Clear query cache for affected document
        for key in list(query_cache):
            query_cache.delete(key)
        logger.info(f"Invalidated cache for {relative_path}")

    def _compute_file_hash(self, file_path):
        """Compute SHA-256 hash of file contents"""
        try:
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            return ""

    def _save_document_hashes(self):
        """Save document hashes to file with timeout"""
        try:
            os.makedirs(os.path.dirname(DOCUMENT_HASH_FILE), exist_ok=True)
            with FileLock(DOCUMENT_HASH_FILE + ".lock", timeout=10):
                with open(DOCUMENT_HASH_FILE, 'w') as f:
                    json.dump(document_hashes, f)
        except Exception as e:
            logger.error(f"Error saving document hashes: {e}")

def start_document_watcher(dir_path):
    """Start watching for document changes"""
    event_handler = DocumentChangeHandler(dir_path)
    observer = Observer()
    observer.schedule(event_handler, path=dir_path, recursive=False)
    observer.start()
    logger.info(f"Started document watcher for {dir_path}")
    return observer

class CachedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """HuggingFace embeddings with diskcache backing store"""
    cache_hits: int = 0
    cache_misses: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        embed_documents with caching while preserving input order.
        """
        keys = [generate_content_hash(text) for text in texts]
        results = [None] * len(texts)
        missing_indices = []
        missing_texts = []
        # collect cached
        for i, key in enumerate(keys):
            if key in embeddings_cache:
                self.cache_hits += 1
                results[i] = embeddings_cache[key]
            else:
                self.cache_misses += 1
                missing_indices.append(i)
                missing_texts.append(texts[i])
        # compute embeddings for missing_texts
        if missing_texts:
            new_embeddings = super().embed_documents(missing_texts)
            for idx, emb in zip(missing_indices, new_embeddings):
                results[idx] = emb
                embeddings_cache[keys[idx]] = emb
        # final sanity check: ensure no None
        for i, r in enumerate(results):
            if r is None:
                # fallback compute single embedding
                emb = super().embed_documents([texts[i]])[0]
                results[i] = emb
                embeddings_cache[keys[i]] = emb
        return results

    def embed_query(self, text: str):
        key = generate_content_hash(text)
        if key in embeddings_cache:
            self.cache_hits += 1
            return embeddings_cache[key]
        self.cache_misses += 1
        embedding = super().embed_query(text)
        embeddings_cache[key] = embedding
        return embedding

    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) if total > 0 else 0.0
        return {"hits": self.cache_hits, "misses": self.cache_misses, "hit_rate": hit_rate}

class CachedRetriever(BaseRetriever):
    """Wrapper around retriever with query caching"""
    base_retriever: BaseRetriever = Field(...)
    cache_hits: int = Field(0)
    cache_misses: int = Field(0)

    def __init__(self, base_retriever):
        super().__init__(base_retriever=base_retriever)

    def get_relevant_documents(self, query):
        key = generate_content_hash(query)
        if key in query_cache:
            cached_entry = query_cache.get(key)
            if time.time() - cached_entry['timestamp'] < CONFIG['CACHE_EXPIRATION']:
                self.cache_hits += 1
                logger.info(f"Using cached query results for: {query[:50]}...")
                return [Document(page_content=doc['page_content'], metadata=doc['metadata']) 
                        for doc in cached_entry['results']]
            else:
                # expired
                try:
                    query_cache.delete(key)
                except Exception:
                    pass

        # cache miss -> fetch
        self.cache_misses += 1
        results = self.base_retriever.get_relevant_documents(query)
        serializable_results = [{'page_content': doc.page_content, 'metadata': doc.metadata} 
                               for doc in results]
        query_cache[key] = {
            'results': serializable_results,
            'timestamp': time.time()
        }
        return results

    def _get_relevant_documents(self, query):
        return self.get_relevant_documents(query)
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total) if total > 0 else 0.0
        return {'hits': self.cache_hits, 'misses': self.cache_misses, 'hit_rate': hit_rate}

def load_embedding_model():
    """Load embedding model with GPU support if available"""
    return CachedHuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1"
    )

def create_vector_db(documents: List[Document], embedding_model):
    """Create and save FAISS vector store"""
    try:
        logger.info(f"[DEBUG] Creating FAISS vector store with {len(documents)} document chunks.")
        db = FAISS.from_documents(documents, embedding_model)
        db.save_local(DB_FAISS_PATH)
        logger.info(f"Created and saved FAISS vector store at {DB_FAISS_PATH}")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise

def initialize_bm25_retriever(documents: List[Document]):
    """Initialize and save BM25 retriever"""
    try:
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = CONFIG['RETRIEVAL_TOP_K']
        with open(BM25_RETRIEVER_PATH, 'wb') as f:
            pickle.dump(bm25_retriever, f)
        logger.info(f"Initialized and saved BM25 retriever at {BM25_RETRIEVER_PATH}")
        return bm25_retriever
    except Exception as e:
        logger.error(f"Error initializing BM25 retriever: {e}")
        raise

def load_bm25_retriever():
    """Load BM25 retriever from disk"""
    if not os.path.exists(BM25_RETRIEVER_PATH):
        raise FileNotFoundError(f"BM25 retriever file not found at {BM25_RETRIEVER_PATH}")
    try:
        with open(BM25_RETRIEVER_PATH, 'rb') as f:
            bm25_retriever = pickle.load(f)
        logger.info("Loaded BM25 retriever")
        return bm25_retriever
    except Exception as e:
        logger.error(f"Error loading BM25 retriever: {e}")
        raise

def _resolve_model_locally(model_id: str, max_retries: int = 3, backoff_seconds: float = 2.0) -> str:
    """Download model snapshot with retries and return local directory. Falls back to cached copy if available."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            local_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=HF_CACHE_DIR,
                local_files_only=False,
                resume_download=True,
            )
            return local_dir
        except Exception as e:
            last_err = e
            time.sleep(backoff_seconds * attempt)
    # Final attempt: try to use cached files only (offline) if present
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=HF_CACHE_DIR,
            local_files_only=True,
            resume_download=True,
        )
        return local_dir
    except Exception:
        if last_err:
            raise last_err
        raise RuntimeError(f"Unable to resolve model {model_id} locally")
    
def build_tinyllama_fallback() -> any:
    """
    Fallback function to load TinyLlama when 7B model fails.
    This is a simplified, reliable version that should work on most systems.
    """
    use_cuda = CONFIG['DEVICE'] == 'cuda' and torch.cuda.is_available()
    fallback_model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    
    logger.info(f"Loading fallback TinyLlama model: {fallback_model_id}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_model_id,
            trust_remote_code=True,
            cache_dir=HF_CACHE_DIR
        )
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if use_cuda:
            # GPU version - lighter quantization for TinyLlama
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model_id,
                torch_dtype=torch.float16,  # Use FP16 for GPU
                device_map="auto",
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
                low_cpu_mem_usage=True
            )
            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=0.1,  # Lower temperature for more consistent output
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            # CPU version
            model = AutoModelForCausalLM.from_pretrained(
                fallback_model_id,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,
                low_cpu_mem_usage=True
            )
            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        logger.info(f"Fallback TinyLlama loaded successfully on {CONFIG['DEVICE']}")
        return gen_pipe
    except Exception as e:
        logger.error(f"Even fallback TinyLlama failed to load: {e}")
        # Fallback: return a mock pipeline for development only
        class MockPipeline:
            def __call__(self, *args, **kwargs):
                return [
                    {"generated_text": "[MOCK OUTPUT] LLM unavailable. Please check model access or download."}
                ]
        logger.warning("Using mock pipeline - for development only!")
        return MockPipeline()

def build_mock_pipeline():
    """
    Ultimate fallback - a mock pipeline for development/testing
    """
    logger.warning("Using mock pipeline - for development only!")
    
    class MockPipeline:
        def __call__(self, prompt, **kwargs):
            return [{
                'generated_text': prompt + " [Mock response: System is in fallback mode. Please check your model configuration.]"
            }]
        
        def __repr__(self):
            return "MockPipeline(development_mode=True)"
    
    return MockPipeline()


def build_generation_pipeline_with_retries():
    """Build pipeline with 7B model and INT4 quantization"""
    use_cuda = CONFIG['DEVICE'] == 'cuda' and torch.cuda.is_available()
    model_id = CONFIG['MODEL_ID']
    
    try:
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        if use_cuda:
            # Configure INT4 quantization
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            # Load model with quantization
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": CONFIG['LLM_DEVICE']},
                quantization_config=quant_config
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=4096,
                max_new_tokens=CONFIG['MAX_NEW_TOKENS']
            )
            logger.info(f"Loaded generation pipeline for {model_id} (device={CONFIG['LLM_DEVICE']})")
            return pipe
        else:
            # CPU fallback (not recommended for 7B)
            logger.warning("7B model on CPU will be very slow!")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},
                trust_remote_code=True
            )
            gen_pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=0.2,
                max_length=4096,
                max_new_tokens=CONFIG['MAX_NEW_TOKENS'],
                device=-1
            )
        return gen_pipe

    except Exception as e:
        logger.error(f"Failed to load 7B model: {e}")
        logger.info("Falling back to TinyLlama...")
        # Fallback to original TinyLlama
        return build_tinyllama_fallback()


from langchain.prompts import PromptTemplate

from langchain.prompts import PromptTemplate

def build_rag_prompt() -> PromptTemplate:
    template = (
        "You are an expert assistant. Answer ONLY with the information from the provided context.\n\n"
        "Rules:\n"
        "- Do NOT reveal or mention the context text itself.\n"
        "- If the context does not contain information that directly answers the question, reply exactly: 'I don't know.'\n"
        "- Never guess, assume, or bring outside knowledge.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Final Answer:"
    )
    return PromptTemplate(template=template, input_variables=["context", "question"])




def process_single_document(file_info):
    """Process a single document file (worker function for parallel processing)"""
    file_path, file_type = file_info
    documents = []
    try:
        if file_type == 'pdf':
            df = extract_text_and_metadata_from_pdf_document(file_path)
        else:  # docx
            df = extract_text_and_metadata_from_docx_document(file_path)
        
        logger.info(f"Extracted text and metadata from {file_path}")
        for index, row in df.iterrows():
            file_name = row.get('Filename', os.path.basename(file_path))
            text = row.get('Text', '')
            page_number = row.get('Page_Number', 0)
            parent_id = row.get('Parent_Id', '')
            doc_id = f"{index}_{parent_id}_{file_name}_{page_number}" if parent_id else f"{index}_{file_name}_{page_number}"
            document = Document(
                page_content=text,
                metadata={
                    'id': doc_id,
                    'type': 'text',
                    'filename': file_name,
                    'page_number': page_number,
                    'content_hash': generate_content_hash(text)
                }
            )
            documents.append(document)
        return documents
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def process_documents_parallel(file_infos):
    """Process multiple documents in parallel using ProcessPoolExecutor"""
    all_documents = []
    try:
        with ProcessPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            futures = [executor.submit(process_single_document, file_info) for file_info in file_infos]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
    except Exception as e:
        logger.error(f"Error setting up parallel processing: {e}")
        raise
    return all_documents

def process_document_file(file_path, file_type, documents):
    """Process a single PDF or DOCX file (sequential)"""
    try:
        if file_type == 'pdf':
            df = extract_text_and_metadata_from_pdf_document(file_path)
        else:  # docx
            df = extract_text_and_metadata_from_docx_document(file_path)
        
        logger.info(f"Extracted text and metadata from {file_path}")
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f'Processing {file_path}'):
            file_name = row.get('Filename', os.path.basename(file_path))
            text = row.get('Text', '')
            page_number = row.get('Page_Number', 0)
            parent_id = row.get('Parent_Id', '')
            doc_id = f"{index}_{parent_id}_{file_name}_{page_number}" if parent_id else f"{index}_{file_name}_{page_number}"
            document = Document(
                page_content=text,
                metadata={
                    'id': doc_id,
                    'type': 'text',
                    'filename': file_name,
                    'page_number': page_number,
                    'content_hash': generate_content_hash(text)
                }
            )
            documents.append(document)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def process_query(query_text: str, qa_chain: RetrievalQA, hallucination_detector: HallucinationDetector) -> Dict:
    """Process a single query with hallucination detection"""
    try:
        # The QA chain expects both 'context' and 'question' as input variables.
        # We provide an empty context here, as the retriever will fill it in.
        try:
            result = qa_chain.invoke({"question": query_text, "context": ""})
        except Exception:
            # fallback
            result = qa_chain({"question": query_text, "context": ""})

        response = result.get('result') or result.get('answer') or result.get('output_text') or str(result)
        source_docs = result.get('source_documents') or result.get('source_documents', []) or []
        if source_docs is None:
            source_docs = []

        is_hallucination, confidence, similar_sentence = hallucination_detector.detect(response, source_docs)

        outputs = [""] * 5
        for j, doc in enumerate(source_docs[:5]):
            outputs[j] = doc.page_content

        return {
            'response': response,
            'outputs': outputs,
            'is_hallucination': is_hallucination,
            'hallucination_confidence': confidence,
            'similar_sentence': similar_sentence if is_hallucination else None
        }
    except Exception as e:
        logger.error(f"Error processing query '{query_text[:50]}...': {e}")
        return {
            'response': "Error processing query",
            'outputs': [""] * 5,
            'is_hallucination': False,
            'hallucination_confidence': 0.0,
            'similar_sentence': None
        }

def process_queries_parallel(test_df: pd.DataFrame, qa_chain: RetrievalQA, hallucination_detector: HallucinationDetector) -> pd.DataFrame:
    """
    Process queries in parallel using ThreadPoolExecutor.
    We use threads because the qa_chain / models are not easily picklable for process workers.
    """
    try:
        # Prepare function wrapper
        def _worker(idx, query_text):
            result = process_query(query_text, qa_chain, hallucination_detector)
            return idx, result

        # Use ThreadPoolExecutor to avoid pickling heavy objects
        with ThreadPoolExecutor(max_workers=CONFIG['MAX_WORKERS']) as executor:
            futures = {executor.submit(_worker, idx, row['Query text']): idx for idx, row in test_df.iterrows()}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing queries"):
                idx = futures[future]
                try:
                    _idx, result = future.result()
                    test_df.at[idx, 'Generated_Response'] = result['response']
                    test_df.at[idx, 'Is_Hallucination'] = result['is_hallucination']
                    test_df.at[idx, 'Hallucination_Confidence'] = result['hallucination_confidence']
                    test_df.at[idx, 'Suggested_Correction'] = result['similar_sentence'] or ""
                    
                    for j in range(5):
                        test_df.at[idx, f'Output_{j+1}'] = result['outputs'][j]
                except Exception as e:
                    logger.error(f"Error processing query at index {idx}: {e}")
                    test_df.at[idx, 'Generated_Response'] = "Error processing query"
        
        return test_df
    except Exception as e:
        logger.error(f"Error in parallel query processing: {e}")
        raise

# --- Encryption Utilities ---
from cryptography.fernet import Fernet
import base64
import os

def get_encryption_key():
    # Use a local file for the key
    key_path = os.getenv('VECTOR_DB_KEY_PATH', 'vectorstore/secret.key')
    if os.path.exists(key_path):
        with open(key_path, 'rb') as f:
            key = f.read()
    else:
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, 'wb') as f:
            f.write(key)
    return key

def encrypt_file(input_path, output_path, key):
    with open(input_path, 'rb') as f:
        data = f.read()
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data)
    with open(output_path, 'wb') as f:
        f.write(encrypted)

def decrypt_file(input_path, output_path, key):
    with open(input_path, 'rb') as f:
        encrypted = f.read()
    fernet = Fernet(key)
    decrypted = fernet.decrypt(encrypted)
    with open(output_path, 'wb') as f:
        f.write(decrypted)

# --- Wrapper for Vector DB Save/Load ---
def save_encrypted_index(index_path, encrypted_path):
    key = get_encryption_key()
    encrypt_file(index_path, encrypted_path, key)

def load_encrypted_index(encrypted_path, decrypted_path):
    key = get_encryption_key()
    decrypt_file(encrypted_path, decrypted_path, key)

# --- Automated Save/Load Encryption in Vector DB Pipeline ---
VECTOR_INDEX_PATH = 'vectorstore/db_faiss/index.faiss'
ENCRYPTED_INDEX_PATH = 'vectorstore/db_faiss/index.faiss.enc'

# --- Multi-Factor Authentication (MFA) Utilities ---
OTP_EXPIRY_SECONDS = 300  # 5 minutes
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'ragforpublicservices@gmail.com'
SMTP_PASSWORD = 'jbwh dqxj yapw piwd'


# Add these fields to your User model in database.py:
# otp_hash = Column(String(255), nullable=True)
# otp_expiry = Column(DateTime, nullable=True)

def generate_and_send_otp(user, db_manager):
    otp = str(secrets.randbelow(1000000)).zfill(6)
    otp_hash = bcrypt.hashpw(otp.encode(), bcrypt.gensalt()).decode()
    expiry = datetime.now() + timedelta(seconds=OTP_EXPIRY_SECONDS)
    # Update user record
    session = db_manager.get_session()
    db_user = session.query(User).filter(User.id == user.id).first()
    db_user.otp_hash = otp_hash
    db_user.otp_expiry = expiry
    session.commit()
    session.close()
    # Send OTP via email
    send_otp_email(user.email, otp)
    return True

def get_sender_email(user):
    # Fetch sender email from database or user profile
    # Example: return user.email or a field in your User model
    return user.email

# Update send_otp_email to use sender from user

def send_otp_email(user, otp):
    sender_email = get_sender_email(user)
    msg = MIMEText(f"Your OTP is: {otp}")
    msg['Subject'] = "Your Login OTP"
    msg['From'] = sender_email
    msg['To'] = user.email
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(msg['From'], [msg['To']], msg.as_string())

def verify_otp(user, otp, db_manager):
    session = db_manager.get_session()
    db_user = session.query(User).filter(User.id == user.id).first()
    valid = False
    if db_user.otp_expiry and datetime.now() <= db_user.otp_expiry:
        valid = bcrypt.checkpw(otp.encode(), db_user.otp_hash.encode())
    session.close()
    return valid



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Process documents and generate responses')
    parser.add_argument('dir_path', type=str, help='Path to directory containing PDF or DOCX files')
    parser.add_argument('csv_path', type=str, help='Path to Test.csv file')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    args = parser.parse_args()

    dir_path = args.dir_path
    csv_path = args.csv_path
    use_parallel = args.parallel

    if not os.path.exists(dir_path):
        logger.error(f"Directory path {dir_path} does not exist")
        sys.exit(1)
    if not os.path.exists(csv_path):
        logger.error(f"CSV path {csv_path} does not exist")
        sys.exit(1)

    observer = start_document_watcher(dir_path)
    embedding_model = load_embedding_model()
    hallucination_detector = HallucinationDetector()

    try:
        # Load or create vector store and BM25 retriever
        documents = []
        rebuild_index = (not os.path.exists(DB_FAISS_PATH)) or (not os.path.exists(BM25_RETRIEVER_PATH))

        if rebuild_index:
            files = [f for f in os.listdir(dir_path) if f.endswith(('.pdf', '.docx'))]
            if use_parallel:
                logger.info("Using parallel document processing")
                file_infos = [(os.path.join(dir_path, file), 'pdf' if file.endswith('.pdf') else 'docx') 
                             for file in files]
                documents = process_documents_parallel(file_infos)
            else:
                logger.info("Using sequential document processing")
                for file in tqdm(files, desc='Processing files'):
                    file_path = os.path.join(dir_path, file)
                    file_type = 'pdf' if file.endswith('.pdf') else 'docx'
                    process_document_file(file_path, file_type, documents)

            documents = deduplicate_documents(documents)
            create_vector_db(documents, embedding_model)
            save_encrypted_index(VECTOR_INDEX_PATH, ENCRYPTED_INDEX_PATH)
            os.remove(VECTOR_INDEX_PATH)  # Remove plaintext after encryption
            bm25_retriever = initialize_bm25_retriever(documents)
        else:
            bm25_retriever = load_bm25_retriever()

        # Load FAISS vector store
        try:
            if os.path.exists(ENCRYPTED_INDEX_PATH):
                load_encrypted_index(ENCRYPTED_INDEX_PATH, VECTOR_INDEX_PATH)
                db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
                os.remove(VECTOR_INDEX_PATH)  # Remove plaintext after loading
            else:
                db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            logger.info("Loaded FAISS vector store")
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {e}")
            raise

        faiss_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": CONFIG['RETRIEVAL_TOP_K'],
                "fetch_k": CONFIG['RETRIEVAL_TOP_K'] * 4,
                "lambda_mult": 0.3
            }
        )
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=CONFIG['SIMILARITY_THRESHOLD'])
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter,
            base_retriever=ensemble_retriever
        )
        cached_retriever = CachedRetriever(compression_retriever)
        dedupe_retriever = DeduplicatingRetriever(cached_retriever)

        # Load language model (LLM)
        try:
            # Load model with device mapping carefully depending on presence of CUDA
            use_cuda = CONFIG['DEVICE'] == 'cuda' and torch.cuda.is_available()
            pipe = build_generation_pipeline_with_retries()
            prompt = build_rag_prompt()
            logger.info(f"Loaded {CONFIG['MODEL_ID']} on device={CONFIG['DEVICE']}")
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            raise

        # Process queries
        try:
            test_df = pd.read_csv(csv_path)
            required_columns = ['Query No', 'Query text', 'Document No', 'Document Title', 'Output_1', 'Output_2', 'Output_3', 'Output_4', 'Output_5']
            if not all(col in test_df.columns for col in required_columns):
                logger.error(f"CSV missing required columns. Required: {required_columns}")
                sys.exit(1)

            # Add columns for hallucination detection
            test_df['Generated_Response'] = ""
            test_df['Is_Hallucination'] = False
            test_df['Hallucination_Confidence'] = 0.0
            test_df['Suggested_Correction'] = ""

            prompt = build_rag_prompt()
            qa_chain = RetrievalQA.from_chain_type(
                llm=HuggingFacePipeline(pipeline=pipe),
                chain_type="stuff",
                retriever=dedupe_retriever,
                chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
                return_source_documents=True
            )

            if use_parallel:
                logger.info("Using parallel query processing (ThreadPoolExecutor)")
                test_df = process_queries_parallel(test_df, qa_chain, hallucination_detector)
            else:
                logger.info("Using sequential query processing")
                for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Processing queries'):
                    query_text = row['Query text']
                    result = process_query(query_text, qa_chain, hallucination_detector)
                    test_df.at[i, 'Generated_Response'] = result['response']
                    test_df.at[i, 'Is_Hallucination'] = result['is_hallucination']
                    test_df.at[i, 'Hallucination_Confidence'] = result['hallucination_confidence']
                    test_df.at[i, 'Suggested_Correction'] = result['similar_sentence'] or ""
                    for j in range(5):
                        test_df.at[i, f'Output_{j+1}'] = result['outputs'][j]

            timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
            file_name = f"output_{timestamp}.csv"
            test_df.to_csv(file_name, index=False)
            logger.info(f"Output saved to {file_name}")

            # Print cache statistics
            try:
                logger.info("\nCache Statistics:")
                logger.info(f"Embeddings Cache: {json.dumps(embedding_model.get_cache_stats(), indent=2)}")
                logger.info(f"Query Cache: {json.dumps(cached_retriever.get_cache_stats(), indent=2)}")
                try:
                    logger.info(f"Embeddings Cache Size: {embeddings_cache.volume()} bytes")
                    logger.info(f"Query Cache Size: {query_cache.volume()} bytes")
                except Exception:
                    logger.debug("Unable to get diskcache volume stats (not supported in this environment)")
            except Exception as e:
                logger.debug(f"Error printing cache stats: {e}")

            # Print hallucination statistics
            if 'Is_Hallucination' in test_df.columns:
                total_queries = len(test_df)
                hallucinations = int(test_df['Is_Hallucination'].sum())
                logger.info(f"\nHallucination Detection Results:")
                logger.info(f"Total queries: {total_queries}")
                logger.info(f"Potential hallucinations detected: {hallucinations} ({(hallucinations/total_queries):.1%})")
                if hallucinations > 0:
                    avg_confidence = float(test_df[test_df['Is_Hallucination']]['Hallucination_Confidence'].mean())
                    logger.info(f"Average confidence of detected hallucinations: {avg_confidence:.2f}")

        except Exception as e:
            logger.error(f"Error processing queries: {e}")
            raise

    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)
    finally:
        try:
            observer.stop()
            observer.join()
        except Exception:
            pass
        try:
            embeddings_cache.close()
            query_cache.close()
        except Exception:
            pass
        logger.info("Cleaned up resources")