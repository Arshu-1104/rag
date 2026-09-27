#!/usr/bin/env python3

import requests
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RAGClient:
    """Python client for the RAG for Public Service Documents API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", token: str = None):
        """
        Initialize the RAG client
        
        Args:
            base_url: Base URL of the API server
            token: JWT access token for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.token = token
        
        if token:
            self.session.headers.update({
                'Authorization': f'Bearer {token}'
            })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make an HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def register_user(self, username: str, email: str, password: str, 
                     full_name: str = None, role: str = "user") -> Dict[str, Any]:
        """
        Register a new user
        
        Args:
            username: Username for the new user
            email: Email address
            password: Password
            full_name: Full name (optional)
            role: User role (default: "user")
        
        Returns:
            User information dictionary
        """
        data = {
            "username": username,
            "email": email,
            "password": password,
            "full_name": full_name,
            "role": role
        }
        
        response = self._make_request('POST', '/auth/register', json=data)
        return response.json()
    
    def login(self, username: str, password: str) -> str:
        """
        Login and get access token
        
        Args:
            username: Username
            password: Password
        
        Returns:
            JWT access token
        """
        data = {
            'username': username,
            'password': password
        }
        
        response = self._make_request('POST', '/auth/login', data=data)
        token_data = response.json()
        
        # Update session with new token
        self.token = token_data['access_token']
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}'
        })
        
        return self.token
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a document
        
        Args:
            file_path: Path to the document file (PDF or DOCX)
        
        Returns:
            Upload response dictionary
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = self._make_request('POST', '/documents/upload', files=files)
        
        return response.json()
    
    def query_documents(self, query: str) -> Dict[str, Any]:
        """
        Query documents using RAG
        
        Args:
            query: The question to ask
        
        Returns:
            Query response with answer and analysis
        """
        data = {
            "query": query
        }
        
        response = self._make_request('POST', '/query', json=data)
        return response.json()
    
    def get_user_documents(self) -> List[Dict[str, Any]]:
        """
        Get documents owned by the current user
        
        Returns:
            List of user documents
        """
        response = self._make_request('GET', '/documents')
        return response.json()
    
    def get_document_changes(self, document_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent changes for a document
        
        Args:
            document_id: ID of the document
            days: Number of days to look back (default: 30)
        
        Returns:
            List of document changes
        """
        params = {'days': days}
        response = self._make_request('GET', f'/documents/{document_id}/changes', params=params)
        return response.json()
    
    def get_user_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent queries by the current user
        
        Args:
            limit: Maximum number of queries to return (default: 50)
        
        Returns:
            List of user queries
        """
        params = {'limit': limit}
        response = self._make_request('GET', '/queries', params=params)
        return response.json()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics for the current user
        
        Returns:
            System statistics dictionary
        """
        response = self._make_request('GET', '/stats')
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health
        
        Returns:
            Health status
        """
        response = self._make_request('GET', '/health')
        return response.json()
    
    def send_document_change_notification(self, document_id: int, change_type: str, 
                                        details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a document change notification (webhook)
        
        Args:
            document_id: ID of the document
            change_type: Type of change (created, updated, deleted)
            details: Additional change details
        
        Returns:
            Webhook response
        """
        data = {
            "document_id": document_id,
            "change_type": change_type,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        
        response = self._make_request('POST', '/webhooks/document-changes', json=data)
        return response.json()

# Example usage and utility functions
def create_sample_user(client: RAGClient, username: str = "test_user") -> str:
    """
    Create a sample user for testing
    
    Args:
        client: RAG client instance
        username: Username for the test user
    
    Returns:
        JWT token for the created user
    """
    try:
        # Register user
        user_data = client.register_user(
            username=username,
            email=f"{username}@example.com",
            password="test_password123",
            full_name=f"Test User {username}",
            role="user"
        )
        print(f"User created: {user_data}")
        
        # Login and get token
        token = client.login(username, "test_password123")
        print(f"Login successful, token: {token[:20]}...")
        
        return token
    
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def upload_sample_documents(client: RAGClient, document_dir: str = "sample_documents"):
    """
    Upload sample documents from a directory
    
    Args:
        client: RAG client instance
        document_dir: Directory containing sample documents
    """
    if not os.path.exists(document_dir):
        print(f"Document directory not found: {document_dir}")
        return
    
    supported_extensions = {'.pdf', '.docx'}
    
    for filename in os.listdir(document_dir):
        file_path = os.path.join(document_dir, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in supported_extensions):
            try:
                result = client.upload_document(file_path)
                print(f"Uploaded {filename}: {result}")
            except Exception as e:
                print(f"Error uploading {filename}: {e}")

def run_sample_queries(client: RAGClient, queries: List[str] = None):
    """
    Run sample queries against the RAG system
    
    Args:
        client: RAG client instance
        queries: List of queries to run (uses defaults if None)
    """
    if queries is None:
        queries = [
            "What are the eligibility criteria for social benefits?",
            "How do I apply for government assistance?",
            "What documents are required for public service applications?",
            "What are the processing times for different services?"
        ]
    
    for query in queries:
        try:
            print(f"\n--- Query: {query} ---")
            result = client.query_documents(query)
            
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence_score']:.2f}")
            print(f"Hallucination: {result['is_hallucination']}")
            print(f"Processing Time: {result['processing_time']:.2f}s")
            print(f"Source Documents: {len(result['source_documents'])}")
            
        except Exception as e:
            print(f"Error running query '{query}': {e}")

def export_user_data(client: RAGClient, output_file: str = "user_data.json"):
    """
    Export all user data to a JSON file
    
    Args:
        client: RAG client instance
        output_file: Output file path
    """
    try:
        # Get all user data
        documents = client.get_user_documents()
        queries = client.get_user_queries(limit=100)
        stats = client.get_system_stats()
        
        # Combine data
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "documents": documents,
            "queries": queries,
            "statistics": stats
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"User data exported to: {output_file}")
        print(f"Documents: {len(documents)}")
        print(f"Queries: {len(queries)}")
        
    except Exception as e:
        print(f"Error exporting user data: {e}")

# Main function for testing
def main():
    """Main function for testing the RAG client"""
    # Initialize client
    client = RAGClient()
    
    print("=== RAG for Public Service Documents - Python Client ===\n")
    
    # Check API health
    try:
        health = client.health_check()
        print(f"API Health: {health}")
    except Exception as e:
        print(f"API not available: {e}")
        return
    
    # Create test user
    token = create_sample_user(client)
    if not token:
        print("Failed to create test user")
        return
    
    # Upload sample documents (if directory exists)
    upload_sample_documents(client)
    
    # Run sample queries
    run_sample_queries(client)
    
    # Export user data
    export_user_data(client)
    
    print("\n=== Client test completed ===")

if __name__ == "__main__":
    main()
