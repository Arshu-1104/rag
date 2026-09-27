# Ingestion/ingest.py

import pandas as pd
import os
from typing import List, Dict, Any
import logging
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx

logger = logging.getLogger(__name__)

def extract_text_and_metadata_from_pdf_document(file_path: str) -> pd.DataFrame:
    """
    Extract text and metadata from PDF document with error handling
    """
    try:
        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            strategy="fast",
        )
        # Ensure all pages are processed
        data = []
        seen_pages = set()
        for i, element in enumerate(elements):
            page_num = getattr(element.metadata, 'page_number', 1) if hasattr(element, 'metadata') else 1
            seen_pages.add(page_num)
            data.append({
                'Filename': os.path.basename(file_path),
                'Text': str(element),
                'Page_Number': page_num,
                'Parent_Id': f"pdf_{i}",
                'Element_Type': str(type(element).__name__)
            })
        # If not all pages detected, fallback to PyPDF2 for missing pages
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        for page_num in range(1, total_pages + 1):
            if page_num not in seen_pages:
                text = reader.pages[page_num - 1].extract_text()
                if text and text.strip():
                    data.append({
                        'Filename': os.path.basename(file_path),
                        'Text': text,
                        'Page_Number': page_num,
                        'Parent_Id': f"pdf_fallback_{page_num}",
                        'Element_Type': 'text'
                    })
        df = pd.DataFrame(data)
        logger.info(f"Successfully processed {file_path}: {len(df)} elements extracted, {total_pages} pages detected")
        return df
    except Exception as e:
        logger.error(f"Error processing PDF {file_path} with unstructured: {e}")
        return extract_text_fallback_pdf(file_path)

def extract_text_fallback_pdf(file_path: str) -> pd.DataFrame:
    """
    Fallback PDF processing using PyPDF2
    """
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        data = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                data.append({
                    'Filename': os.path.basename(file_path),
                    'Text': text,
                    'Page_Number': page_num + 1,
                    'Parent_Id': f"pdf_fallback_{page_num}",
                    'Element_Type': 'text'
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Fallback processing successful for {file_path}: {len(df)} pages processed")
        return df
        
    except Exception as e:
        logger.error(f"Fallback PDF processing failed for {file_path}: {e}")
        # Return empty DataFrame with correct structure
        return pd.DataFrame(columns=['Filename', 'Text', 'Page_Number', 'Parent_Id', 'Element_Type'])

def extract_text_and_metadata_from_docx_document(file_path: str) -> pd.DataFrame:
    """
    Extract text and metadata from DOCX document
    """
    try:
        elements = partition_docx(
            filename=file_path,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
        )
        
        data = []
        for i, element in enumerate(elements):
            data.append({
                'Filename': os.path.basename(file_path),
                'Text': str(element),
                'Page_Number': 1,  # DOCX doesn't have traditional pages
                'Parent_Id': f"docx_{i}",
                'Element_Type': str(type(element).__name__)
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Successfully processed DOCX {file_path}: {len(df)} elements extracted")
        return df
        
    except Exception as e:
        logger.error(f"Error processing DOCX {file_path}: {e}")
        return extract_text_fallback_docx(file_path)

def extract_text_fallback_docx(file_path: str) -> pd.DataFrame:
    """
    Fallback DOCX processing using python-docx
    """
    try:
        from docx import Document
        
        doc = Document(file_path)
        data = []
        
        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text:  # Only add non-empty paragraphs
                data.append({
                    'Filename': os.path.basename(file_path),
                    'Text': text,
                    'Page_Number': 1,
                    'Parent_Id': f"docx_fallback_{i}",
                    'Element_Type': 'paragraph'
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Fallback DOCX processing successful for {file_path}: {len(df)} paragraphs processed")
        return df
        
    except Exception as e:
        logger.error(f"Fallback DOCX processing failed for {file_path}: {e}")
        return pd.DataFrame(columns=['Filename', 'Text', 'Page_Number', 'Parent_Id', 'Element_Type'])
