# LegalAI - Multi-User Legal Document Processing & AI Assistant

A sophisticated legal support system that combines TiDB vector search, advanced document processing, and specialized AI agents to provide personalized legal assistance with user-specific document isolation.

## 🏗️ Architecture Overview

![Architecture diagram](https://github.com/Clean-earthw/legalai/blob/main/docs/architecture.jpg)

The system consists of three main layers:

1. **Frontend Layer**: React/Next.js web application
2. **API Layer**: FastAPI REST endpoints with user authentication
3. **Data Layer**: TiDB with vector embeddings and document storage

## 🔧 Backend Architecture & Features

### TiDB Vector Database Integration

The system uses TiDB as a distributed SQL database with vector search capabilities for storing and retrieving document chunks.

#### Database Schema

```python
class UserDocumentChunk(TableModel):
    __tablename__ = "user_semantic_embeddings"
    
    id: int = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)  # User isolation
    session_id: Optional[str] = Field(default=None, index=True)  # Session tracking
    text: str = Field(sa_column=Column(TEXT))
    document_id: str = Field(index=True)
    document_name: str = Field(sa_column=Column(TEXT))
    document_type: str = Field(default="text")
    page_number: Optional[int] = Field(default=None)
    embedding: List[float] = self.embedding_fn.VectorField(source_field="text")
    section: Optional[str] = Field(sa_column=Column(TEXT), default=None)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    file_size: Optional[int] = Field(default=None)
    chunk_index: Optional[int] = Field(default=None)
```

#### Connection Setup

```python
def init_clients():
    """Initialize TiDB client and embedding function."""
    # Initialize TiDB client
    tidb_client = TiDBClient.connect(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT", 4000)),
        username=os.getenv("TIDB_USERNAME"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
        ensure_db=True,
    )
    
    # Initialize embedding function (Jina or OpenAI)
    if embedding_provider == "jina":
        embedding_fn = EmbeddingFunction(
            model_name="jina_ai/jina-embeddings-v4",
            api_key=os.getenv("JINA_API_KEY"),
            timeout=30
        )
    
    return tidb_client, embedding_fn
```

### Document Chunk Processing

Documents are processed into searchable chunks with metadata preservation:

```python
def add_document_from_text(self, user_id: str, text: str, document_name: str, 
                          session_id: Optional[str] = None):
    """Add a text document to the store with user isolation."""
    
    # Clean and normalize text
    cleaned_text = self._clean_text(text)
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(cleaned_text)
    
    # Create document chunk objects with embeddings
    documents = []
    for i, chunk in enumerate(chunks):
        doc_chunk = self.DocumentChunk(
            user_id=user_id,
            session_id=session_id,
            text=cleaned_chunk,
            document_id=document_id,
            document_name=cleaned_document_name,
            section=self._identify_section(cleaned_chunk),
            chunk_index=i,
        )
        documents.append(doc_chunk)
    
    # Insert with automatic embedding generation
    return self._insert_documents(documents)
```

### Hybrid Search Implementation

The system implements hybrid search combining vector similarity with metadata filtering:

```python
def search(self, user_id: str, query: str, limit: int = 3, 
          session_id: Optional[str] = None):
    """Search with user isolation and hybrid filtering."""
    
    # Vector similarity search
    search_query = self.table.search(query, search_type="vector").limit(limit)
    
    # Add user isolation filters
    filters = {"user_id": {"$eq": user_id}}
    if session_id:
        filters["session_id"] = {"$eq": session_id}
    
    search_query = search_query.filter(filters)
    
    # Apply reranking if available
    if self.reranker:
        search_query = search_query.rerank(self.reranker, "text")
    
    return search_query.to_list()
```

### AI Agent Routing System

The system uses specialized AI agents for different legal domains:

```python
class AgentName(str, Enum):
    EMPLOYMENT = "Employment Expert"
    COMPLIANCE = "Compliance Specialist" 
    EQUITY = "Equity Management Expert"

def process_query(self, user_id: str, query: str, session_id: Optional[str] = None):
    """Route query to appropriate agent with user context."""
    
    # Route query to appropriate agent
    routing_decision = self._route_query(query)
    
    # Get user-specific context
    relevant_context = self.get_user_relevant_context(user_id, query, session_id)
    
    # Process with specialized agent
    agent_handlers = {
        AgentName.EMPLOYMENT: self._handle_employment_query,
        AgentName.COMPLIANCE: self._handle_compliance_query,
        AgentName.EQUITY: self._handle_equity_query
    }
    
    return agent_handlers[routing_decision.agent_name](user_id, query, session_id)
```

## 🚀 Backend Setup & Installation

### Prerequisites

- Python 3.8+
- TiDB Serverless Account
- Jina AI API Key
- OpenAI API Key (optional)

### Environment Variables

Create a `.env` file in the root directory:

```bash
# TiDB Configuration
TIDB_HOST=gateway01.ap-southeast-1.prod.aws.tidbcloud.com
TIDB_PORT=4000
TIDB_USERNAME=your_username
TIDB_PASSWORD=your_password
TIDB_DATABASE=your_database

# AI Configuration
JINA_API_KEY=your_jina_api_key
GEMINI_API_KEY=your_gemini_api_key
AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEN_AI_MODEL=gemini-1.5-flash

# Embedding Configuration
EMBEDDING_PROVIDER=jina
JINA_AI_API_KEY=your_jina_api_key
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Clean-earthw/legalai.git
cd legalai
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install optional dependencies for document processing**:
```bash
# For PDF processing
pip install PyMuPDF PyPDF2

# For DOCX processing  
pip install python-docx docx2txt
```

4. **Initialize the database**:
```bash
python -c "from agents.rag.document_store import init_clients, UserDocumentStore; tidb, emb = init_clients(); store = UserDocumentStore(tidb, emb)"
```

### Running the Backend

**Development Mode**:
```bash
python main.py
```

**Production Mode**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

### API Endpoints

#### Core Endpoints

**Health Check**:
```bash
curl -X GET "http://localhost:8000/health"
```

**Upload Document**:
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "X-User-Id: user123" \
  -H "X-Session-Id: session456" \
  -F "file=@document.pdf"
```

**Ask Question**:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{"question": "What are the employment policies?"}'
```

**Search Documents**:
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{"query": "contract terms", "limit": 5}'
```

#### Document Management

**Get User Documents**:
```bash
curl -X GET "http://localhost:8000/documents" \
  -H "X-User-Id: user123"
```

**Delete Document**:
```bash
curl -X DELETE "http://localhost:8000/documents/{document_id}" \
  -H "X-User-Id: user123"
```

## 🖥️ Frontend Setup & Usage

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

1. **Navigate to frontend directory**:
```bash
cd frontend
```

2. **Install dependencies**:
```bash
npm install
# or
yarn install
```

3. **Configure environment variables**:
```bash
# Create .env.local file
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=LegalAI
```

4. **Run development server**:
```bash
npm run dev
# or  
yarn dev
```

The application will be available at `http://localhost:3000`

### Frontend Features

[Screenshots will be added here showing:]

1. **Login/User Selection Screen**
   - User ID input
   - Session management

2. **Document Upload Interface** 
   - Drag & drop file upload
   - Support for PDF, DOCX, TXT files
   - Upload progress indicators

3. **Chat Interface**
   - Real-time question answering
   - Document source citations
   - Agent routing indicators

4. **Document Management Dashboard**
   - List of uploaded documents
   - Search and filter capabilities
   - Delete and organize functions

5. **Search Interface**
   - Advanced document search
   - Similarity scoring
   - Preview of relevant chunks

## 📊 Key Technical Features

### Vector Search Performance

- **Embedding Model**: Jina AI v4 (1024 dimensions)
- **Search Type**: Cosine similarity with L2 normalization
- **Reranking**: Jina Reranker for improved relevance
- **Response Time**: <200ms for typical queries

### Document Processing Pipeline

```python
Document Upload → Text Extraction → Text Cleaning → Chunking → Embedding Generation → Vector Storage → Index Creation
```

### User Isolation Architecture

- Every document chunk tagged with `user_id`
- All queries filtered by user context
- Session-based document organization
- No cross-user data leakage

### Scalability Features

- TiDB handles distributed storage automatically
- Horizontal scaling through multiple API instances  
- Async processing for large document uploads
- Connection pooling for database efficiency

## 🔧 Configuration Options

### Text Splitting Configuration

```python
# Recursive character splitter (default)
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " "],
    chunk_size=1000,
    chunk_overlap=100,
)

# Character splitter (alternative)
character_splitter = CharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
```

### Embedding Provider Options

```python
# Jina AI (recommended)
embedding_fn = EmbeddingFunction(
    model_name="jina_ai/jina-embeddings-v4",
    api_key=os.getenv("JINA_API_KEY"),
    timeout=30
)

# OpenAI (alternative)
embedding_fn = EmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
```

## 🧪 Testing

### Backend Tests

```bash
# Run unit tests
pytest tests/

# Test specific module
pytest tests/test_document_store.py

# Test with coverage
pytest --cov=agents tests/
```

### API Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with sample data
python scripts/test_api.py
```

## 📈 Monitoring & Debugging

### Health Check Details

The `/health` endpoint provides comprehensive system status:

```json
{
  "status": "healthy",
  "components": {
    "document_store": "healthy",
    "embedding_function": "connected", 
    "reranker": "initialized",
    "database": "connected"
  }
}
```

### Logging Configuration

```python
# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 🔐 Security Considerations

### Current Implementation
- Header-based user identification
- User data isolation at database level
- Input validation and sanitization
- Temporary file cleanup

### Production Recommendations
- Implement proper authentication (JWT tokens)
- Add rate limiting per user
- Use HTTPS in production
- Add input size restrictions
- Implement audit logging

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  legalai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TIDB_HOST=${TIDB_HOST}
      - TIDB_USERNAME=${TIDB_USERNAME}
      - TIDB_PASSWORD=${TIDB_PASSWORD}
    volumes:
      - ./logs:/app/logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
- Create an issue in the [GitHub repository](https://github.com/Clean-earthw/legalai/issues)
- Check the documentation and examples
- Review the API documentation at `/docs` endpoint

## 🔮 Future Roadmap

- [ ] Multi-language support
- [ ] Advanced document analytics
- [ ] Integration with legal databases
- [ ] Mobile application
- [ ] Enterprise SSO integration
- [ ] Advanced role-based permissions