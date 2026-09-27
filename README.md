# RAG for Public Service Documents

A comprehensive Retrieval-Augmented Generation (RAG) system designed specifically for public service document analysis, with advanced features for user management, document versioning, performance monitoring, and export capabilities.

## 🚀 Features

### Core RAG Capabilities
- **Advanced Document Processing**: Support for PDF and DOCX files with intelligent text extraction
- **Multi-Modal Retrieval**: Combines FAISS vector search with BM25 for optimal document retrieval
- **Hallucination Detection**: Built-in system to detect and flag potential AI hallucinations
- **MMR Retrieval**: Maximum Marginal Relevance for diverse and relevant context selection
- **Caching System**: Intelligent caching for embeddings and query results

### User Management & Access Control
- **Multi-User Support**: User registration, authentication, and role-based access
- **JWT Authentication**: Secure token-based authentication system
- **User Roles**: Admin, User, and Viewer roles with different permissions
- **Session Management**: Track user activity and login history

### Document Management & Versioning
- **Document Versioning**: Track document changes and maintain version history
- **Change Detection**: Automatic detection of document modifications
- **Audit Trail**: Complete audit log of all document operations
- **Document Ownership**: Users can only access their own documents

### Export & Integration Features
- **Multiple Export Formats**: PDF, Markdown, JSON, and CSV exports
- **Professional Reports**: Generate comprehensive PDF reports with analysis
- **API Integration**: RESTful API for integration with other systems
- **Webhook Support**: Real-time notifications for document changes

### Performance Optimizations
- **Background Processing**: Asynchronous document processing with Celery
- **Performance Monitoring**: Real-time system metrics and performance analysis
- **Prometheus Metrics**: Standardized metrics for monitoring and alerting
- **Redis Caching**: High-performance caching for improved response times
- **Optimized Retrieval**: MMR-based retrieval for better context selection

### Modern UI
- **Dark Theme**: Professional dark-themed interface
- **Three-Column Layout**: Clean, organized workspace design
- **Real-time Analysis**: Live confidence scoring and hallucination detection
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- CUDA-compatible GPU (optional, for faster processing)

### Dependencies
All dependencies are listed in `requirements.txt`:

```bash
# Core RAG dependencies
faiss-cpu
huggingface_hub
langchain
langchain-community
langchain-core
numpy
pandas
torch
transformers
sentence-transformers

# Document processing
unstructured
pdf2image
pdfminer.six
opencv-python-headless
pillow

# Web framework and UI
streamlit
fastapi
uvicorn

# Database and caching
sqlalchemy
psycopg2-binary
redis
diskcache

# Authentication and security
python-jose[cryptography]
passlib[bcrypt]
python-multipart

# Performance monitoring
prometheus-client
psutil
celery
structlog

# Export capabilities
reportlab
weasyprint

# Utilities
tqdm
python-dotenv
pydantic-settings
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag-public-service-documents
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Initialize the database**:
```bash
python -c "from database import get_db_manager; get_db_manager()"
```

## 🚀 Quick Start

### Option 1: Streamlit UI (Recommended for users)
```bash
streamlit run ui.py
```

### Option 2: API Server (For developers and integrations)
```bash
python api.py
```

### Option 3: Command Line Interface
```bash
python rag.py <document_directory> <query_csv_file> --parallel
```

## 📖 Usage Guide

### 1. User Registration and Login

#### Via API:
```bash
# Register a new user
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "secure_password",
    "full_name": "John Doe",
    "role": "user"
  }'

# Login and get access token
curl -X POST "http://localhost:8000/auth/login" \
  -F "username=john_doe" \
  -F "password=secure_password"
```

#### Via Streamlit UI:
1. Open the application in your browser
2. Use the document upload panel to initialize the system
3. Upload your first documents

### 2. Document Upload and Processing

#### Via API:
```bash
# Upload a document
curl -X POST "http://localhost:8000/documents/upload" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@document.pdf"
```

#### Via Streamlit UI:
1. Click "Initialize System" if not already done
2. Use the file uploader in the left panel
3. Select PDF or DOCX files
4. Click "Process Documents"

### 3. Querying Documents

#### Via API:
```bash
# Query documents
curl -X POST "http://localhost:8000/query" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the eligibility criteria for social benefits?"
  }'
```

#### Via Streamlit UI:
1. Enter your question in the center panel
2. Click "Ask Question"
3. View the response with confidence analysis

### 4. Exporting Results

#### Via API:
```bash
# Get user statistics
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# Get user queries
curl -X GET "http://localhost:8000/queries" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### Via Streamlit UI:
1. Use the download buttons in the query panel
2. Export individual responses as Markdown
3. Share responses via generated links

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Database
DATABASE_URL=sqlite:///./rag_system.db
# For PostgreSQL: postgresql://user:password@localhost/rag_db

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Redis (for caching and background tasks)
REDIS_URL=redis://localhost:6379
CELERY_BROKER_URL=redis://localhost:6379/0

# Model Configuration
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DEVICE=cuda  # or cpu

# Performance
MAX_WORKERS=4
BATCH_SIZE=32
```

### Model Configuration

Edit `rag.py` to customize model settings:

```python
CONFIG = {
    'MODEL_ID': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'MAX_NEW_TOKENS': 512,
    'RETRIEVAL_TOP_K': 6,
    'SIMILARITY_THRESHOLD': 0.76,
    'TOP_P': 0.9,
    'REPETITION_PENALTY': 1.05
}
```

## 📊 Performance Monitoring

### Prometheus Metrics

The system exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Available metrics:
- `rag_queries_total`: Total queries processed
- `rag_query_duration_seconds`: Query processing time
- `rag_hallucinations_total`: Hallucinations detected
- `rag_system_memory_bytes`: Memory usage
- `rag_system_cpu_percent`: CPU usage

### Performance Dashboard

Access performance insights via the API:

```bash
curl -X GET "http://localhost:8000/stats" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🔒 Security Features

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: Bcrypt password hashing
- **Role-Based Access**: Different permissions for different user roles
- **Input Validation**: Comprehensive input validation and sanitization
- **Audit Logging**: Complete audit trail of all operations

## 📈 Scaling and Deployment

### Production Deployment

1. **Use PostgreSQL** instead of SQLite:
```env
DATABASE_URL=postgresql://user:password@localhost/rag_db
```

2. **Set up Redis** for caching and background tasks:
```env
REDIS_URL=redis://your-redis-server:6379
CELERY_BROKER_URL=redis://your-redis-server:6379/0
```

3. **Configure Celery workers**:
```bash
celery -A performance_monitor.celery_app worker --loglevel=info
```

4. **Use a production WSGI server**:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 API Documentation

Once the API server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc

## 📝 Export Formats

### PDF Reports
- Professional formatting with tables and charts
- Complete analysis including confidence scores
- Source document citations
- User activity reports

### Markdown
- Clean, readable format
- Perfect for documentation
- Includes all analysis data
- Source citations with previews

### JSON
- Machine-readable format
- Complete metadata
- Structured data for integrations
- API-friendly format

### CSV
- Spreadsheet-compatible
- Bulk data analysis
- Query history export
- Performance metrics

## 🔄 Document Versioning

The system automatically tracks document changes:

1. **Version History**: Each document maintains a complete version history
2. **Change Detection**: Automatic detection when documents are modified
3. **Audit Trail**: Complete log of who changed what and when
4. **Rollback Capability**: Ability to revert to previous versions

## 📊 Performance Optimization

### Caching Strategy
- **Embedding Cache**: Caches document embeddings for faster retrieval
- **Query Cache**: Caches frequent queries and responses
- **Redis Integration**: High-performance caching with Redis

### Background Processing
- **Async Document Processing**: Non-blocking document upload and processing
- **Celery Tasks**: Background task processing for heavy operations
- **Queue Management**: Intelligent task queuing and prioritization

### Monitoring and Alerts
- **Real-time Metrics**: Live system performance monitoring
- **Performance Recommendations**: Automated suggestions for optimization
- **Resource Usage Tracking**: Memory, CPU, and disk usage monitoring

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Check internet connection for model download
   - Verify sufficient disk space
   - Try different model in configuration

2. **Memory Issues**:
   - Reduce batch size in configuration
   - Use CPU instead of GPU
   - Increase system memory

3. **Database Errors**:
   - Check database connection string
   - Ensure database is initialized
   - Verify user permissions

4. **Performance Issues**:
   - Enable Redis for caching
   - Use background processing for large documents
   - Monitor system resources

### Logs

Check logs for detailed error information:
- Application logs: `pipeline.log`
- System logs: Check your system's log files
- API logs: Available in the console when running the API server

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with LangChain and Hugging Face Transformers
- Uses FAISS for vector similarity search
- Streamlit for the web interface
- FastAPI for the REST API
- Prometheus for metrics collection

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the troubleshooting section

---

**RAG for Public Service Documents** - Making document analysis accessible, secure, and efficient for public service organizations.
