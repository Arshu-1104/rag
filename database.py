#!/usr/bin/env python3

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from pydantic import BaseModel
import logging
from sqlalchemy.orm import declarative_base
logger = logging.getLogger(__name__)

Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    role = Column(String(20), default="user")  # admin, user, viewer
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime)
    # MFA fields
    otp_hash = Column(String(255), nullable=True)
    otp_expiry = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="owner")
    queries = relationship("QueryLog", back_populates="user")

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False)  # SHA-256 hash
    file_type = Column(String(10), nullable=False)  # pdf, docx
    file_size = Column(Integer)
    version = Column(Integer, default=1)
    parent_id = Column(Integer, ForeignKey("documents.id"), nullable=True)  # For versioning
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    status = Column(String(20), default="active")  # active, archived, deleted
    doc_metadata = Column(JSON)  # Store document metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    versions = relationship("Document", backref="parent", remote_side="Document.id")
    # Removed chunks relationship

    # DocumentChunk model removed

class QueryLog(Base):
    __tablename__ = "query_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text, nullable=False)
    confidence_score = Column(Integer)  # 0-100
    is_hallucination = Column(Boolean, default=False)
    processing_time = Column(Integer)  # milliseconds
    source_documents = Column(JSON)  # List of document IDs used
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="queries")

    # DocumentChange model removed

# Pydantic Models for API
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None
    role: str = "user"

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    version: int
    status: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class QueryLogResponse(BaseModel):
    id: int
    query_text: str
    response_text: str
    confidence_score: Optional[int]
    is_hallucination: bool
    processing_time: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Database Manager
class DatabaseManager:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "sqlite:///rag_system.db"
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self):
        return self.SessionLocal()

    def create_user(self, user_data: dict) -> Optional[User]:
        session = self.get_session()
        try:
            # Check if user exists
            if session.query(User).filter((User.username == user_data["username"]) | (User.email == user_data["email"])).first():
                return None
            hashed_pw = user_data["password"]
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                hashed_password=hashed_pw,
                full_name=user_data.get("full_name"),
                role=user_data.get("role", "user"),
                is_active=True
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating user: {e}")
            return None
        finally:
            session.close()

    def get_user_by_username(self, username: str) -> Optional[User]:
        session = self.get_session()
        try:
            return session.query(User).filter(User.username == username).first()
        finally:
            session.close()

    def verify_user_password(self, username: str, password: str) -> Optional[User]:
        import bcrypt
        user = self.get_user_by_username(username)
        if user and bcrypt.checkpw(password.encode(), user.hashed_password.encode()):
            return user
        return None
    
    def create_document(self, filename: str, file_path: str, file_hash: str, 
                       file_type: str, owner_id: int, file_size: int = None,
                       metadata: Dict = None) -> Document:
        """Create a new document record"""
        with self.get_session() as session:
            doc = Document(
                filename=filename,
                file_path=file_path,
                file_hash=file_hash,
                file_type=file_type,
                file_size=file_size,
                owner_id=owner_id,
                doc_metadata=metadata or {}
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc
    
    def get_document_by_hash(self, file_hash: str) -> Optional[Document]:
        """Get document by file hash"""
        with self.get_session() as session:
            return session.query(Document).filter(Document.file_hash == file_hash).first()
    
    def update_document_version(self, document_id: int, new_hash: str, 
                               changed_by: int) -> Document:
        """Create a new version of a document"""
        with self.get_session() as session:
            old_doc = session.query(Document).filter(Document.id == document_id).first()
            if not old_doc:
                raise ValueError("Document not found")
            
            # Create new version
            new_doc = Document(
                filename=old_doc.filename,
                file_path=old_doc.file_path,
                file_hash=new_hash,
                file_type=old_doc.file_type,
                file_size=old_doc.file_size,
                owner_id=old_doc.owner_id,
                version=old_doc.version + 1,
                parent_id=old_doc.id,
                doc_metadata=old_doc.doc_metadata
            )
            session.add(new_doc)
            
            # Archive old version
            old_doc.status = "archived"
            
            # Log the change
            # Removed DocumentChange creation
            
            session.commit()
            session.refresh(new_doc)
            return new_doc
    
    def log_query(self, user_id: int, query_text: str, response_text: str,
                  confidence_score: int = None, is_hallucination: bool = False,
                  processing_time: int = None, source_documents: List[int] = None) -> QueryLog:
        """Log a query and response"""
        with self.get_session() as session:
            log = QueryLog(
                user_id=user_id,
                query_text=query_text,
                response_text=response_text,
                confidence_score=confidence_score,
                is_hallucination=is_hallucination,
                processing_time=processing_time,
                source_documents=source_documents or []
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log
    
    def get_user_documents(self, user_id: int, status: str = "active") -> List[Document]:
        """Get documents owned by a user"""
        with self.get_session() as session:
            return session.query(Document).filter(
                Document.owner_id == user_id,
                Document.status == status
            ).order_by(Document.updated_at.desc()).all()
    
    # Removed get_document_changes method
    
    def get_recent_queries(self, user_id: int = None, limit: int = 50) -> List[QueryLog]:
        """Get recent queries, optionally filtered by user"""
        with self.get_session() as session:
            query = session.query(QueryLog)
            if user_id:
                query = query.filter(QueryLog.user_id == user_id)
            return query.order_by(QueryLog.created_at.desc()).limit(limit).all()

def create_all_tables(database_url: str = None):
    """Create all tables in the database."""
    url = database_url or "sqlite:///rag_system.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)

# Global database instance
db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        database_url = os.getenv("DATABASE_URL", "sqlite:///./rag_system.db")
        db_manager = DatabaseManager(database_url)
    return db_manager

if __name__ == "__main__":
    # Create all tables if running as script
    create_all_tables()
