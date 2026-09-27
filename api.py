#!/usr/bin/env python3

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
from database import (
    DatabaseManager, UserCreate, UserResponse, DocumentResponse, QueryLogResponse,
    get_db_manager
)
from rag_core import RAGSystem
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="RAG for Public Service Documents API",
    description="API for document management and querying with RAG capabilities",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

# Pydantic models for API
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    is_hallucination: bool
    source_documents: List[Dict]
    processing_time: float

class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str

class DocumentChangeNotification(BaseModel):
    document_id: int
    change_type: str
    timestamp: datetime
    details: Dict

# Global RAG system instance
rag_system = None
executor = ThreadPoolExecutor(max_workers=4)

def get_rag_system():
    """Get the global RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
        rag_system.initialize_system()
    return rag_system

# Authentication functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data

def get_current_user(token_data: TokenData = Depends(verify_token)):
    db_manager = get_db_manager()
    user = db_manager.get_user_by_username(token_data.username)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# API Endpoints

@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """Register a new user"""
    db_manager = get_db_manager()
    
    # Check if user already exists
    existing_user = db_manager.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    try:
        user = db_manager.create_user(user_data)
        return UserResponse.from_orm(user)
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating user"
        )

@app.post("/auth/login", response_model=Token)
async def login(username: str = Form(...), password: str = Form(...)):
    """Login and get access token"""
    db_manager = get_db_manager()
    user = db_manager.verify_user_password(username, password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    # Update last login
    with db_manager.get_session() as session:
        user.last_login = datetime.utcnow()
        session.commit()
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """Upload a document"""
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and DOCX files are supported"
        )
    
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Calculate file hash
        with open(temp_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        db_manager = get_db_manager()
        
        # Check if document already exists
        existing_doc = db_manager.get_document_by_hash(file_hash)
        if existing_doc:
            return DocumentUploadResponse(
                document_id=existing_doc.id,
                filename=file.filename,
                status="exists",
                message="Document already exists in the system"
            )
        
        # Create document record
        file_type = "pdf" if file.filename.lower().endswith('.pdf') else "docx"
        doc = db_manager.create_document(
            filename=file.filename,
            file_path=temp_path,
            file_hash=file_hash,
            file_type=file_type,
            owner_id=current_user.id,
            file_size=os.path.getsize(temp_path)
        )
        
        # Process document with RAG system
        rag = get_rag_system()
        
        # Convert UploadFile to format expected by RAG system
        class MockUploadFile:
            def __init__(self, path, filename, content_type):
                self.path = path
                self.name = filename
                self.type = content_type
            
            def getbuffer(self):
                with open(self.path, 'rb') as f:
                    return f.read()
        
        mock_file = MockUploadFile(temp_path, file.filename, file.content_type)
        
        # Process in background
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            executor, 
            rag.add_documents, 
            [mock_file], 
            True
        )
        
        if stats.get("errors"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {stats['errors']}"
            )
        
        return DocumentUploadResponse(
            document_id=doc.id,
            filename=file.filename,
            status="success",
            message=f"Document uploaded and processed successfully. Added {stats.get('new_documents', 0)} segments."
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    query_request: QueryRequest,
    current_user = Depends(get_current_user)
):
    """Query documents using RAG"""
    rag = get_rag_system()
    
    if not rag.initialized or not rag.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents available for querying"
        )
    
    try:
        # Process query
        start_time = datetime.now()
        response_data = rag.answer_query(query_request.query)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log query
        db_manager = get_db_manager()
        source_doc_ids = [doc.metadata.get('id', 0) for doc in response_data.get('source_documents', [])]
        
        db_manager.log_query(
            user_id=current_user.id,
            query_text=query_request.query,
            response_text=response_data['answer'],
            confidence_score=int((1 - response_data.get('hallucination_info', {}).get('confidence', 0)) * 100),
            is_hallucination=response_data.get('hallucination_info', {}).get('is_hallucination', False),
            processing_time=int(processing_time * 1000),
            source_documents=source_doc_ids
        )
        
        return QueryResponse(
            answer=response_data['answer'],
            confidence_score=1 - response_data.get('hallucination_info', {}).get('confidence', 0),
            is_hallucination=response_data.get('hallucination_info', {}).get('is_hallucination', False),
            source_documents=[
                {
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "page_number": doc.metadata.get('page_number', 0),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                for doc in response_data.get('source_documents', [])
            ],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/documents", response_model=List[DocumentResponse])
async def get_user_documents(current_user = Depends(get_current_user)):
    """Get documents owned by the current user"""
    db_manager = get_db_manager()
    documents = db_manager.get_user_documents(current_user.id)
    return [DocumentResponse.from_orm(doc) for doc in documents]

@app.get("/documents/{document_id}/changes")
async def get_document_changes(
    document_id: int,
    days: int = 30,
    current_user = Depends(get_current_user)
):
    """Get recent changes for a document"""
    db_manager = get_db_manager()
    changes = db_manager.get_document_changes(document_id, days)
    
    return [
        {
            "id": change.id,
            "change_type": change.change_type,
            "timestamp": change.created_at,
            "details": change.change_details
        }
        for change in changes
    ]

@app.get("/queries", response_model=List[QueryLogResponse])
async def get_user_queries(
    limit: int = 50,
    current_user = Depends(get_current_user)
):
    """Get recent queries by the current user"""
    db_manager = get_db_manager()
    queries = db_manager.get_recent_queries(current_user.id, limit)
    return [QueryLogResponse.from_orm(query) for query in queries]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/stats")
async def get_system_stats(current_user = Depends(get_current_user)):
    """Get system statistics"""
    rag = get_rag_system()
    db_manager = get_db_manager()
    
    # Get user's documents
    user_docs = db_manager.get_user_documents(current_user.id)
    
    # Get user's recent queries
    recent_queries = db_manager.get_recent_queries(current_user.id, 100)
    
    # Calculate statistics
    total_queries = len(recent_queries)
    hallucination_count = sum(1 for q in recent_queries if q.is_hallucination)
    avg_processing_time = sum(q.processing_time for q in recent_queries) / total_queries if total_queries > 0 else 0
    
    return {
        "user_documents": len(user_docs),
        "total_queries": total_queries,
        "hallucination_rate": (hallucination_count / total_queries * 100) if total_queries > 0 else 0,
        "avg_processing_time_ms": avg_processing_time,
        "system_documents": len(rag.documents) if rag.documents else 0
    }

# Webhook endpoint for document change notifications
@app.post("/webhooks/document-changes")
async def document_change_webhook(notification: DocumentChangeNotification):
    """Webhook endpoint for document change notifications"""
    # In a real implementation, you would:
    # 1. Verify the webhook signature
    # 2. Send notifications to subscribed users
    # 3. Update any cached data
    
    logger.info(f"Document change notification: {notification}")
    
    # Example: Send email notification to document owner
    # send_document_change_email(notification)
    
    return {"status": "notification_received"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
