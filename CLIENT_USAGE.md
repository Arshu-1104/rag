# RAG Python Client Usage Guide

This guide shows how to use the Python client for the RAG for Public Service Documents API, converting all curl commands into Python code.

## Installation

First, install the required dependencies:

```bash
pip install requests
```

Or install all project dependencies:

```bash
pip install -r requirements.txt
```

## Basic Setup

```python
from rag_client import RAGClient

# Initialize client
client = RAGClient(base_url="http://localhost:8000")
```

## Authentication

### Register a New User

**Curl equivalent:**
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password",
    "full_name": "John Doe",
    "role": "user"
  }'
```

**Python code:**
```python
user_data = client.register_user(
    username="john_doe",
    email="john@example.com",
    password="secure_password",
    full_name="John Doe",
    role="user"
)
print(f"User created: {user_data}")
```

### Login and Get Access Token

**Curl equivalent:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -F "username=john_doe" \
  -F "password=secure_password"
```

**Python code:**
```python
token = client.login("john_doe", "secure_password")
print(f"Access token: {token[:20]}...")
```

## Document Operations

### Upload a Document

**Curl equivalent:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@document.pdf"
```

**Python code:**
```python
# Make sure you're logged in first
client.login("john_doe", "secure_password")

# Upload document
result = client.upload_document("path/to/document.pdf")
print(f"Upload result: {result}")
```

### Get User Documents

**Curl equivalent:**
```bash
curl -X GET "http://localhost:8000/documents" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Python code:**
```python
documents = client.get_user_documents()
for doc in documents:
    print(f"Document: {doc['filename']} (v{doc['version']})")
```

### Get Document Changes

**Curl equivalent:**
```bash
curl -X GET "http://localhost:8000/documents/123/changes?days=30" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Python code:**
```python
changes = client.get_document_changes(document_id=123, days=30)
for change in changes:
    print(f"Change: {change['change_type']} at {change['timestamp']}")
```

## Query Operations

### Query Documents

**Curl equivalent:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the eligibility criteria for social benefits?"
  }'
```

**Python code:**
```python
result = client.query_documents("What are the eligibility criteria for social benefits?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']:.2f}")
print(f"Hallucination: {result['is_hallucination']}")
print(f"Processing Time: {result['processing_time']:.2f}s")
print(f"Source Documents: {len(result['source_documents'])}")
```

### Get User Queries

**Curl equivalent:**
```bash
curl -X GET "http://localhost:8000/queries?limit=50" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Python code:**
```python
queries = client.get_user_queries(limit=50)
for query in queries:
    print(f"Query: {query['query_text'][:50]}...")
    print(f"Confidence: {query['confidence_score']}%")
```

## System Operations

### Get System Statistics

**Curl equivalent:**
```bash
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

**Python code:**
```python
stats = client.get_system_stats()
print(f"User Documents: {stats['user_documents']}")
print(f"Total Queries: {stats['total_queries']}")
print(f"Hallucination Rate: {stats['hallucination_rate']:.1f}%")
print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.0f}ms")
```

### Health Check

**Curl equivalent:**
```bash
curl http://localhost:8000/health
```

**Python code:**
```python
health = client.health_check()
print(f"API Status: {health}")
```

### Send Document Change Notification

**Curl equivalent:**
```bash
curl -X POST "http://localhost:8000/webhooks/document-changes" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": 123,
    "change_type": "updated",
    "timestamp": "2024-01-01T12:00:00",
    "details": {"version": 2}
  }'
```

**Python code:**
```python
result = client.send_document_change_notification(
    document_id=123,
    change_type="updated",
    details={"version": 2}
)
print(f"Notification sent: {result}")
```

## Complete Example Workflow

```python
from rag_client import RAGClient
import os

def complete_workflow():
    """Complete workflow example"""
    
    # 1. Initialize client
    client = RAGClient()
    
    # 2. Check API health
    try:
        health = client.health_check()
        print(f"✅ API is running: {health}")
    except Exception as e:
        print(f"❌ API not available: {e}")
        return
    
    # 3. Register and login
    try:
        user_data = client.register_user(
            username="test_user",
            email="test@example.com",
            password="password123",
            full_name="Test User"
        )
        print(f"✅ User registered: {user_data['username']}")
        
        token = client.login("test_user", "password123")
        print(f"✅ Login successful")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return
    
    # 4. Upload documents
    document_dir = "documents"
    if os.path.exists(document_dir):
        for filename in os.listdir(document_dir):
            if filename.lower().endswith(('.pdf', '.docx')):
                file_path = os.path.join(document_dir, filename)
                try:
                    result = client.upload_document(file_path)
                    print(f"✅ Uploaded {filename}: {result['status']}")
                except Exception as e:
                    print(f"❌ Failed to upload {filename}: {e}")
    
    # 5. Query documents
    queries = [
        "What are the eligibility criteria?",
        "How do I apply for benefits?",
        "What documents are required?"
    ]
    
    for query in queries:
        try:
            result = client.query_documents(query)
            print(f"\n🔍 Query: {query}")
            print(f"Answer: {result['answer'][:100]}...")
            print(f"Confidence: {result['confidence_score']:.2f}")
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # 6. Get statistics
    try:
        stats = client.get_system_stats()
        print(f"\n📊 Statistics:")
        print(f"Documents: {stats['user_documents']}")
        print(f"Queries: {stats['total_queries']}")
        print(f"Hallucination Rate: {stats['hallucination_rate']:.1f}%")
    except Exception as e:
        print(f"❌ Statistics failed: {e}")

if __name__ == "__main__":
    complete_workflow()
```

## Error Handling

The client includes comprehensive error handling:

```python
from rag_client import RAGClient

client = RAGClient()

try:
    # Try to access protected endpoint without authentication
    documents = client.get_user_documents()
except Exception as e:
    print(f"Expected error (no auth): {e}")

try:
    # Try to upload non-existent file
    result = client.upload_document("non_existent.pdf")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Upload error: {e}")

try:
    # Try to query with empty string
    result = client.query_documents("")
except Exception as e:
    print(f"Query error: {e}")
```

## Batch Operations

For batch operations, you can use the utility functions:

```python
from rag_client import RAGClient, upload_sample_documents, run_sample_queries, export_user_data

client = RAGClient()
client.login("username", "password")

# Upload all documents from a directory
upload_sample_documents(client, "documents/")

# Run multiple queries
queries = [
    "What are the eligibility criteria?",
    "How do I apply?",
    "What documents are needed?"
]
run_sample_queries(client, queries)

# Export all user data
export_user_data(client, "my_data.json")
```

## Advanced Usage

### Custom Headers

```python
client = RAGClient()
client.session.headers.update({
    'Custom-Header': 'Custom-Value'
})
```

### Timeout Configuration

```python
client = RAGClient()
client.session.timeout = 30  # 30 seconds timeout
```

### Retry Logic

```python
import time
from requests.exceptions import RequestException

def query_with_retry(client, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.query_documents(query)
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
```

## Integration Examples

### Integration with Streamlit

```python
import streamlit as st
from rag_client import RAGClient

@st.cache_resource
def get_client():
    return RAGClient()

def main():
    st.title("RAG Client Integration")
    
    client = get_client()
    
    # Login form
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            try:
                token = client.login(username, password)
                st.success("Login successful!")
            except Exception as e:
                st.error(f"Login failed: {e}")
    
    # Query form
    with st.form("query"):
        query = st.text_area("Enter your question")
        submitted = st.form_submit_button("Ask")
        
        if submitted and query:
            try:
                result = client.query_documents(query)
                st.write("**Answer:**", result['answer'])
                st.write("**Confidence:**", f"{result['confidence_score']:.2f}")
            except Exception as e:
                st.error(f"Query failed: {e}")

if __name__ == "__main__":
    main()
```

### Integration with Jupyter Notebook

```python
# In a Jupyter notebook
from rag_client import RAGClient
import pandas as pd

# Initialize client
client = RAGClient()
client.login("username", "password")

# Get user data
documents = client.get_user_documents()
queries = client.get_user_queries(limit=100)

# Create DataFrames for analysis
docs_df = pd.DataFrame(documents)
queries_df = pd.DataFrame(queries)

# Analyze query performance
print("Query Performance Analysis:")
print(queries_df.groupby('is_hallucination')['confidence_score'].mean())
print(queries_df['processing_time'].describe())
```

This Python client provides a clean, Pythonic interface to your RAG API, making it easy to integrate into your applications and scripts!
