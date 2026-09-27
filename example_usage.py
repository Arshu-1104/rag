#!/usr/bin/env python3

"""
Example usage of the RAG Client for Public Service Documents

This script demonstrates how to use the Python client to interact with the RAG API.
Make sure the API server is running before executing this script.
"""

from rag_client import RAGClient
import os

def example_basic_usage():
    """Basic usage example"""
    print("=== Basic RAG Client Usage ===\n")
    
    # Initialize client
    client = RAGClient()
    
    # 1. Check if API is running
    try:
        health = client.health_check()
        print(f"✅ API is running: {health}")
    except Exception as e:
        print(f"❌ API not available: {e}")
        return
    
    # 2. Register a new user
    try:
        user_data = client.register_user(
            username="john_doe",
            email="john@example.com",
            password="secure_password123",
            full_name="John Doe",
            role="user"
        )
        print(f"✅ User registered: {user_data['username']}")
    except Exception as e:
        print(f"❌ User registration failed: {e}")
        return
    
    # 3. Login and get token
    try:
        token = client.login("john_doe", "secure_password123")
        print(f"✅ Login successful, token: {token[:20]}...")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # 4. Upload a document (if available)
    sample_doc = "sample_document.pdf"
    if os.path.exists(sample_doc):
        try:
            result = client.upload_document(sample_doc)
            print(f"✅ Document uploaded: {result['message']}")
        except Exception as e:
            print(f"❌ Document upload failed: {e}")
    else:
        print(f"⚠️  Sample document not found: {sample_doc}")
    
    # 5. Query documents
    try:
        query = "What are the eligibility criteria for social benefits?"
        result = client.query_documents(query)
        
        print(f"\n✅ Query Results:")
        print(f"Question: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence_score']:.2f}")
        print(f"Hallucination Detected: {result['is_hallucination']}")
        print(f"Processing Time: {result['processing_time']:.2f}s")
        print(f"Source Documents: {len(result['source_documents'])}")
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
    
    # 6. Get user statistics
    try:
        stats = client.get_system_stats()
        print(f"\n✅ User Statistics:")
        print(f"Documents: {stats['user_documents']}")
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Hallucination Rate: {stats['hallucination_rate']:.1f}%")
        print(f"Avg Processing Time: {stats['avg_processing_time_ms']:.0f}ms")
        
    except Exception as e:
        print(f"❌ Statistics failed: {e}")

def example_batch_operations():
    """Example of batch operations"""
    print("\n=== Batch Operations Example ===\n")
    
    # Initialize client with existing token
    client = RAGClient()
    
    # Login
    try:
        token = client.login("john_doe", "secure_password123")
        print("✅ Logged in for batch operations")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # Upload multiple documents
    document_dir = "documents"
    if os.path.exists(document_dir):
        print(f"📁 Uploading documents from {document_dir}:")
        for filename in os.listdir(document_dir):
            if filename.lower().endswith(('.pdf', '.docx')):
                file_path = os.path.join(document_dir, filename)
                try:
                    result = client.upload_document(file_path)
                    print(f"  ✅ {filename}: {result['status']}")
                except Exception as e:
                    print(f"  ❌ {filename}: {e}")
    
    # Run multiple queries
    queries = [
        "What are the eligibility criteria for social benefits?",
        "How do I apply for government assistance?",
        "What documents are required for public service applications?",
        "What are the processing times for different services?",
        "How can I check my application status?"
    ]
    
    print(f"\n🔍 Running {len(queries)} queries:")
    for i, query in enumerate(queries, 1):
        try:
            result = client.query_documents(query)
            print(f"  {i}. {query[:50]}...")
            print(f"     Confidence: {result['confidence_score']:.2f}, "
                  f"Hallucination: {result['is_hallucination']}")
        except Exception as e:
            print(f"  {i}. ❌ Error: {e}")

def example_data_export():
    """Example of data export operations"""
    print("\n=== Data Export Example ===\n")
    
    # Initialize client
    client = RAGClient()
    
    # Login
    try:
        token = client.login("john_doe", "secure_password123")
        print("✅ Logged in for data export")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # Get user documents
    try:
        documents = client.get_user_documents()
        print(f"📄 User Documents ({len(documents)}):")
        for doc in documents:
            print(f"  - {doc['filename']} (v{doc['version']}, {doc['status']})")
    except Exception as e:
        print(f"❌ Failed to get documents: {e}")
    
    # Get user queries
    try:
        queries = client.get_user_queries(limit=10)
        print(f"\n🔍 Recent Queries ({len(queries)}):")
        for query in queries:
            print(f"  - {query['query_text'][:50]}...")
            print(f"    Confidence: {query['confidence_score']}%, "
                  f"Time: {query['processing_time']}ms")
    except Exception as e:
        print(f"❌ Failed to get queries: {e}")
    
    # Export all data
    try:
        from rag_client import export_user_data
        export_user_data(client, "my_rag_data.json")
        print("✅ Data exported to my_rag_data.json")
    except Exception as e:
        print(f"❌ Export failed: {e}")

def example_error_handling():
    """Example of error handling"""
    print("\n=== Error Handling Example ===\n")
    
    client = RAGClient()
    
    # Try to access protected endpoint without authentication
    try:
        result = client.get_user_documents()
        print("✅ Got user documents")
    except Exception as e:
        print(f"❌ Expected error (no auth): {e}")
    
    # Try to upload non-existent file
    try:
        result = client.upload_document("non_existent_file.pdf")
        print("✅ Uploaded file")
    except Exception as e:
        print(f"❌ Expected error (file not found): {e}")
    
    # Try to query with empty string
    try:
        result = client.query_documents("")
        print("✅ Query successful")
    except Exception as e:
        print(f"❌ Expected error (empty query): {e}")

if __name__ == "__main__":
    print("RAG for Public Service Documents - Python Client Examples")
    print("=" * 60)
    
    # Run examples
    example_basic_usage()
    example_batch_operations()
    example_data_export()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("\nTo use the client in your own code:")
    print("1. from rag_client import RAGClient")
    print("2. client = RAGClient()")
    print("3. client.register_user(...) or client.login(...)")
    print("4. client.query_documents('your question')")
