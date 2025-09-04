import os
import uuid
import logging
import re
import dotenv
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field, Column
from pytidb.datatype import TEXT
from pytidb.embeddings import EmbeddingFunction
from pytidb.rerankers import Reranker
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from datetime import datetime

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("document_store")

# Set Jina API key
os.environ["JINA_AI_API_KEY"] = os.getenv("JINA_API_KEY")


def init_clients():
    """Initialize TiDB client and embedding function."""
    # Validate required environment variables
    required_vars = ["TIDB_HOST", "TIDB_USERNAME", "TIDB_PASSWORD", "TIDB_DATABASE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    # Initialize TiDB client
    tidb_client = TiDBClient.connect(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT", 4000)),
        username=os.getenv("TIDB_USERNAME"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
        ensure_db=True,
    )
    
    # Initialize embedding function based on provider
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "jina").lower()
    #tidb_client.configure_embedding_provider("jina", api_key=os.getenv(""))
    
    if embedding_provider == "openai":
        embedding_fn = EmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif embedding_provider == "jina":
        embedding_fn = EmbeddingFunction(
            model_name="jina_ai/jina-embeddings-v4",
            api_key=os.getenv("JINA_API_KEY"),
            timeout=30
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    return tidb_client, embedding_fn


class UserDocumentStore:
    """Enhanced document store with user-specific document isolation and session management."""
    
    def __init__(self, tidb_client, embedding_fn):
        self.tidb_client = tidb_client
        self.embedding_fn = embedding_fn
        self.table = None
        self.DocumentChunk = None
        
        # Initialize text splitters
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " "],
            chunk_size=1000,
            chunk_overlap=100,
        )
        
        self.character_splitter = CharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        
        # Initialize reranker for Jina
        self.reranker = None
        if "jina" in embedding_fn.model_name.lower():
            try:
                self.reranker = Reranker(model_name="jina_ai/jina-reranker-m0")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
        
        # Create the table and schema
        self._initialize_table()

    def _initialize_table(self):
        """Initialize the user-specific document chunk table and schema."""
        try:
            # Create the UserDocumentChunk class dynamically with user isolation
            class UserDocumentChunk(TableModel):
                __tablename__ = "user_semantic_embeddings"
                __table_args__ = {"extend_existing": True}

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
                file_size: Optional[int] = Field(default=None)  # Track file size
                chunk_index: Optional[int] = Field(default=None)  # Track chunk order

            self.DocumentChunk = UserDocumentChunk
            self.table = self.tidb_client.create_table(schema=UserDocumentChunk, if_exists="skip")
            logger.info("User document table initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize table: {e}")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text to handle encoding issues."""
        if not text:
            return ""
            
        try:
            # Handle different encodings
            if isinstance(text, bytes):
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text = text.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    text = text.decode('utf-8', errors='replace')
            
            # Ensure text is string
            text = str(text)
            
            # Replace problematic Unicode characters
            replacements = {
                '\u2018': "'", '\u2019': "'",  # Smart quotes
                '\u201c': '"', '\u201d': '"',  # Smart double quotes
                '\u2013': '-', '\u2014': '--', # Dashes
                '\u2026': '...', '\u00a0': ' ', # Ellipsis, non-breaking space
            }
            
            for old_char, new_char in replacements.items():
                text = text.replace(old_char, new_char)
            
            # Remove non-printable characters but keep newlines and tabs
            text = ''.join(char for char in text if char.isprintable() or char in '\n\t\r ')
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {e}")
            return str(text).encode('ascii', errors='ignore').decode('ascii')

    def _identify_section(self, text: str) -> Optional[str]:
        """Extract section information from text."""
        if not text:
            return None
            
        section_patterns = [
            r'^\s*(\d+(?:\.\d+)*)\s+(.+?)(?=\n|$)',  # Numbered sections
            r'^\s*(Chapter|Section|Part)\s+(\d+[.\d]*)\s*:?\s*(.+?)(?=\n|$)',  # Named sections
            r'^\s*([A-Z][A-Z\s]{2,20})(?=\n)',  # All caps headers
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return f"{match.group(1)} {match.group(2)}"
                else:
                    return match.group(1)
        
        return None

    def add_document_from_text(
            self, 
            user_id: str,
            text: str, 
            document_name: str, 
            session_id: Optional[str] = None,
            document_id: Optional[str] = None,
            document_type: str = "text",
            splitter_type: str = "recursive",
            page_number: Optional[int] = None
            ) -> Dict[str, Any]:
        """Add a text document to the store with user isolation."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not text or not text.strip():
            raise ValueError("Text content cannot be empty")
        
        if not document_name or not document_name.strip():
            raise ValueError("Document name cannot be empty")
            
        if not document_id:
            document_id = str(uuid.uuid4())
            
        try:
            # Clean the input text and document name
            cleaned_text = self._clean_text(text)
            cleaned_document_name = self._clean_text(document_name)
            
            if not cleaned_text.strip():
                raise ValueError("Document contains no valid text after cleaning")
            
            # Choose splitter and create chunks
            splitter = self.recursive_splitter if splitter_type == "recursive" else self.character_splitter
            chunks = splitter.split_text(cleaned_text)
            
            if not chunks:
                logger.warning("No chunks created from document")
                return {"document_id": document_id, "chunks_added": 0}
            
            logger.info(f"Created {len(chunks)} chunks for document: {cleaned_document_name} (User: {user_id})")
            
            # Create document chunk objects
            documents = []
            for i, chunk in enumerate(chunks):
                cleaned_chunk = self._clean_text(chunk)
                if not cleaned_chunk.strip():
                    continue
                    
                doc_chunk = self.DocumentChunk(
                    user_id=user_id,
                    session_id=session_id,
                    text=cleaned_chunk,
                    document_id=document_id,
                    document_name=cleaned_document_name,
                    document_type=document_type,
                    page_number=page_number,
                    section=self._identify_section(cleaned_chunk),
                    file_size=len(text.encode('utf-8')),
                    chunk_index=i,
                )
                documents.append(doc_chunk)
            
            if not documents:
                raise ValueError("No valid chunks created after cleaning")
            
            # Insert documents with multiple fallback methods
            chunks_added = self._insert_documents(documents)
            
            logger.info(f"Successfully added {chunks_added} chunks for document {cleaned_document_name} (User: {user_id})")
            return {
                "document_id": document_id, 
                "chunks_added": chunks_added,
                "user_id": user_id,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error adding document from text: {e}")
            raise Exception(f"Failed to add document '{document_name}': {str(e)}")

    def _insert_documents(self, documents: List) -> int:
        """Insert documents with multiple fallback methods."""
        if not documents:
            return 0
            
        insertion_methods = [
            ("individual_insert", self._insert_individual),
            ("bulk_insert", self._insert_bulk),
            ("dict_insert", self._insert_as_dicts),
        ]
        
        for method_name, method_func in insertion_methods:
            try:
                logger.info(f"Trying insertion method: {method_name}")
                result = method_func(documents)
                logger.info(f"Method {method_name} succeeded, inserted {result} documents")
                return result
            except Exception as e:
                logger.warning(f"Method {method_name} failed: {e}")
                continue
        
        raise Exception("All document insertion methods failed")

    def _insert_individual(self, documents: List) -> int:
        """Insert documents one by one."""
        count = 0
        for doc in documents:
            self.table.insert(doc)
            count += 1
        return count

    def _insert_bulk(self, documents: List) -> int:
        """Insert documents in bulk."""
        self.table.insert(*documents)
        return len(documents)

    def _insert_as_dicts(self, documents: List) -> int:
        """Insert documents as dictionaries."""
        doc_dicts = []
        for doc in documents:
            if hasattr(doc, 'dict'):
                doc_dict = doc.dict()
            else:
                doc_dict = {k: v for k, v in doc.__dict__.items() if not k.startswith('_')}
            doc_dicts.append(doc_dict)
        
        for doc_dict in doc_dicts:
            self.table.insert(doc_dict)
        
        return len(doc_dicts)

    def add_document_from_file(
        self, 
        user_id: str,
        file_path: str, 
        document_name: Optional[str] = None,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None,
        splitter_type: str = "recursive"
    ) -> Dict[str, Any]:
        """Add a document from text file with user isolation."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in ['.txt', '.md']:
            raise ValueError(f"Unsupported file type: {file_extension}. Only .txt and .md files are supported.")

        try:
            # Try multiple encodings
            text = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                raise ValueError("Could not decode file with any supported encoding")
            
            return self.add_document_from_text(
                user_id=user_id,
                text=text, 
                document_name=document_name or file_path.stem, 
                session_id=session_id,
                document_id=document_id, 
                document_type="text",
                splitter_type=splitter_type
            )
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            raise ValueError(f"Failed to load file: {str(e)}")

    def search(
        self, 
        user_id: str,
        query: str, 
        limit: int = 3, 
        session_id: Optional[str] = None,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
        with_scores: bool = False
    ) -> Union[List[Dict[str, Any]], List[Tuple]]:
        """Search for documents using vector similarity search with user isolation."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not query or not query.strip():
            return [] if not with_scores else []
            
        try:
            cleaned_query = self._clean_text(query)
            
            if not cleaned_query.strip():
                logger.warning("Query is empty after cleaning")
                return [] if not with_scores else []
            
            # Build search query with user isolation
            search_query = self.table.search(cleaned_query, search_type="vector").limit(limit)
            
            # Add mandatory user filter using proper syntax
            filters = {"user_id": {"$eq": user_id}}
            
            # Add optional filters
            if session_id:
                filters["session_id"] = {"$eq": session_id}
            if document_type:
                filters["document_type"] = {"$eq": document_type}
            if document_id:
                filters["document_id"] = {"$eq": document_id}
            
            search_query = search_query.filter(filters)
            
            # Add reranking if available
            if self.reranker:
                try:
                    search_query = search_query.rerank(self.reranker, "text")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
            
            results = search_query.to_list()
            
            if with_scores:
                scored_results = []
                for result in results:
                    score = result.get('_distance', 0.0)
                    similarity_score = max(0.0, 1.0 - score)  # Convert distance to similarity
                    scored_results.append((result, similarity_score))
                return scored_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search for user {user_id}: {e}")
            return [] if not with_scores else []

    def search_all_documents(
        self, 
        query: str, 
        limit: int = 3, 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        document_type: Optional[str] = None,
        document_id: Optional[str] = None,
        with_scores: bool = False
    ) -> Union[List[Dict[str, Any]], List[Tuple]]:
        """Search for documents across all users using vector similarity search."""
        
        if not query or not query.strip():
            return [] if not with_scores else []
            
        try:
            cleaned_query = self._clean_text(query)
            
            if not cleaned_query.strip():
                logger.warning("Query is empty after cleaning")
                return [] if not with_scores else []
            
            # Build search query without user isolation
            search_query = self.table.search(cleaned_query, search_type="vector").limit(limit)
            
            # Add optional filters using proper syntax
            filters = {}
            if user_id:
                filters["user_id"] = {"$eq": user_id}
            if session_id:
                filters["session_id"] = {"$eq": session_id}
            if document_type:
                filters["document_type"] = {"$eq": document_type}
            if document_id:
                filters["document_id"] = {"$eq": document_id}
            
            if filters:
                search_query = search_query.filter(filters)
            
            # Add reranking if available
            if self.reranker:
                try:
                    search_query = search_query.rerank(self.reranker, "text")
                except Exception as e:
                    logger.warning(f"Reranking failed: {e}")
            
            results = search_query.to_list()
            
            if with_scores:
                scored_results = []
                for result in results:
                    score = result.get('_distance', 0.0)
                    similarity_score = max(0.0, 1.0 - score)  # Convert distance to similarity
                    scored_results.append((result, similarity_score))
                return scored_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error during global search: {e}")
            return [] if not with_scores else []

    def similarity_search_with_score(
        self, 
        user_id: str,
        query: str, 
        k: int = 3, 
        session_id: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> List[Tuple]:
        """Search for similar documents and return with scores."""
        return self.search(user_id, query, limit=k, session_id=session_id, 
                          document_type=document_type, with_scores=True)

    def delete_document(self, user_id: str, document_id: str) -> Dict[str, Any]:
        """Delete all chunks with the given document_id for a specific user."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
            
        try:
            # Get chunks count before deletion
            chunks_count = self._count_document_chunks(user_id, document_id)
            
            if chunks_count == 0:
                logger.warning(f"No chunks found with document_id: {document_id} for user: {user_id}")
                return {"document_id": document_id, "chunks_deleted": 0, "user_id": user_id}
            
            # Delete chunks with user isolation using proper filter syntax
            self.table.delete(filters={"user_id": {"$eq": user_id}, "document_id": {"$eq": document_id}})
            
            logger.info(f"Deleted {chunks_count} chunks with document_id: {document_id} for user: {user_id}")
            return {"document_id": document_id, "chunks_deleted": chunks_count, "user_id": user_id}

        except Exception as e:
            logger.error(f"Error deleting documents from TiDB: {e}")
            raise Exception(f"Failed to delete document {document_id} for user {user_id}: {str(e)}")

    def delete_document_global(self, document_id: str) -> Dict[str, Any]:
        """Delete all chunks with the given document_id across all users."""
        
        if not document_id or not document_id.strip():
            raise ValueError("Document ID cannot be empty")
            
        try:
            # Get chunks count before deletion
            chunks_count = self._count_global_document_chunks(document_id)
            
            if chunks_count == 0:
                logger.warning(f"No chunks found with document_id: {document_id}")
                return {"document_id": document_id, "chunks_deleted": 0}
            
            # Delete chunks globally using proper filter syntax
            self.table.delete(filters={"document_id": {"$eq": document_id}})
            
            logger.info(f"Deleted {chunks_count} chunks with document_id: {document_id} globally")
            return {"document_id": document_id, "chunks_deleted": chunks_count}

        except Exception as e:
            logger.error(f"Error deleting document globally from TiDB: {e}")
            raise Exception(f"Failed to delete document {document_id} globally: {str(e)}")

    def _count_document_chunks(self, user_id: str, document_id: str) -> int:
        """Count chunks for a specific document ID and user."""
        try:
            # Use proper filter syntax
            filters = {"user_id": {"$eq": user_id}, "document_id": {"$eq": document_id}}
            
            # Try multiple methods to count chunks with user isolation
            count_methods = [
                lambda: len(self.tidb_client.query("SELECT COUNT(*) FROM user_semantic_embeddings WHERE user_id = :user_id AND document_id = :document_id", {"user_id": user_id, "document_id": document_id}).to_list()),
                lambda: len(self.table.query(filters).to_list()),
                lambda: len([c for c in self.table.scan().to_list() 
                           if c.get('user_id') == user_id and c.get('document_id') == document_id]),
            ]
            
            for method in count_methods:
                try:
                    return method()
                except Exception:
                    continue
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error counting chunks for document {document_id} and user {user_id}: {e}")
            return 0

    def _count_global_document_chunks(self, document_id: str) -> int:
        """Count chunks for a specific document ID across all users."""
        try:
            # Use proper filter syntax
            filters = {"document_id": {"$eq": document_id}}
            
            # Try multiple methods to count chunks globally
            count_methods = [
                lambda: len(self.tidb_client.query("SELECT COUNT(*) FROM user_semantic_embeddings WHERE document_id = :document_id", {"document_id": document_id}).to_list()),
                lambda: len(self.table.query(filters).to_list()),
                lambda: len([c for c in self.table.scan().to_list() 
                           if c.get('document_id') == document_id]),
            ]
            
            for method in count_methods:
                try:
                    return method()
                except Exception:
                    continue
            
            return 0
            
        except Exception as e:
            logger.warning(f"Error counting chunks for document {document_id} globally: {e}")
            return 0

    def get_user_documents(self, user_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all unique documents for a specific user."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        try:
            if self.table is None:
                logger.error("Table not initialized")
                return []
            
            # Verify table exists
            try:
                tables = self.tidb_client.list_tables()
                if "user_semantic_embeddings" not in tables:
                    logger.error("user_semantic_embeddings table does not exist")
                    return []
            except Exception as e:
                logger.error(f"Failed to verify table existence: {e}")
                return []
            
            # Try multiple methods to fetch user's chunks
            all_chunks = self._fetch_user_chunks(user_id, session_id)
            
            if not all_chunks:
                logger.info(f"No chunks found for user: {user_id}")
                return []
            
            logger.info(f"Retrieved {len(all_chunks)} total chunks for user: {user_id}")
            
            # Group by document_id to get unique documents
            documents_map = {}
            processed_chunks = 0
            
            for chunk in all_chunks:
                try:
                    # Handle different chunk formats (dict vs object)
                    doc_data = self._extract_chunk_data(chunk)
                    
                    if not doc_data['document_id']:
                        logger.warning(f"Chunk missing document_id, skipping")
                        continue
                    
                    doc_id = doc_data['document_id']
                    
                    if doc_id not in documents_map:
                        documents_map[doc_id] = {
                            'document_id': doc_id,
                            'document_name': doc_data['document_name'],
                            'document_type': doc_data['document_type'],
                            'user_id': doc_data['user_id'],
                            'session_id': doc_data.get('session_id'),
                            'created_at': doc_data.get('created_at'),
                            'file_size': doc_data.get('file_size'),
                            'chunks_count': 0
                        }
                    
                    documents_map[doc_id]['chunks_count'] += 1
                    processed_chunks += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            documents = list(documents_map.values())
            
            logger.info(f"Processed {processed_chunks} chunks into {len(documents)} unique documents for user: {user_id}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents for user {user_id}: {e}", exc_info=True)
            raise Exception(f"Failed to retrieve documents for user {user_id}: {str(e)}")

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from all users and sessions."""
        
        try:
            if self.table is None:
                logger.error("Table not initialized")
                return []
            
            # Verify table exists
            try:
                tables = self.tidb_client.list_tables()
                if "user_semantic_embeddings" not in tables:
                    logger.error("user_semantic_embeddings table does not exist")
                    return []
            except Exception as e:
                logger.error(f"Failed to verify table existence: {e}")
                return []
            
            # Try multiple methods to fetch all chunks
            all_chunks = self._fetch_all_chunks()
            
            if not all_chunks:
                logger.info("No chunks found in the system")
                return []
            
            logger.info(f"Retrieved {len(all_chunks)} total chunks from all users")
            
            # Group by document_id to get unique documents
            documents_map = {}
            processed_chunks = 0
            
            for chunk in all_chunks:
                try:
                    # Handle different chunk formats (dict vs object)
                    doc_data = self._extract_chunk_data(chunk)
                    
                    if not doc_data['document_id']:
                        logger.warning(f"Chunk missing document_id, skipping")
                        continue
                    
                    doc_id = doc_data['document_id']
                    
                    if doc_id not in documents_map:
                        documents_map[doc_id] = {
                            'document_id': doc_id,
                            'document_name': doc_data['document_name'],
                            'document_type': doc_data['document_type'],
                            'user_id': doc_data['user_id'],
                            'session_id': doc_data.get('session_id'),
                            'created_at': doc_data.get('created_at'),
                            'file_size': doc_data.get('file_size'),
                            'chunks_count': 0
                        }
                    
                    documents_map[doc_id]['chunks_count'] += 1
                    processed_chunks += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing chunk: {e}")
                    continue
            
            documents = list(documents_map.values())
            
            # Sort documents by created_at (newest first) if available
            documents.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            logger.info(f"Processed {processed_chunks} chunks into {len(documents)} unique documents from all users")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}", exc_info=True)
            raise Exception(f"Failed to retrieve all documents: {str(e)}")

    def get_documents_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all documents for a specific user (improved version of get_user_documents)."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        try:
            # Get all documents first
            all_documents = self.get_all_documents()
            
            # Filter by user_id
            user_documents = [doc for doc in all_documents if doc.get('user_id') == user_id]
            
            logger.info(f"Found {len(user_documents)} documents for user: {user_id}")
            
            return user_documents
            
        except Exception as e:
            logger.error(f"Error getting documents for user {user_id}: {e}")
            raise Exception(f"Failed to retrieve documents for user {user_id}: {str(e)}")

    def get_documents_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all documents for a specific session."""
        
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        
        try:
            # Get all documents first
            all_documents = self.get_all_documents()
            
            # Filter by session_id
            session_documents = [doc for doc in all_documents if doc.get('session_id') == session_id]
            
            logger.info(f"Found {len(session_documents)} documents for session: {session_id}")
            
            return session_documents
            
        except Exception as e:
            logger.error(f"Error getting documents for session {session_id}: {e}")
            raise Exception(f"Failed to retrieve documents for session {session_id}: {str(e)}")

    def get_documents_by_type(self, document_type: str) -> List[Dict[str, Any]]:
        """Retrieve all documents of a specific type."""
        
        if not document_type or not document_type.strip():
            raise ValueError("Document type cannot be empty")
        
        try:
            # Get all documents first
            all_documents = self.get_all_documents()
            
            # Filter by document_type
            typed_documents = [doc for doc in all_documents if doc.get('document_type') == document_type]
            
            logger.info(f"Found {len(typed_documents)} documents of type: {document_type}")
            
            return typed_documents
            
        except Exception as e:
            logger.error(f"Error getting documents of type {document_type}: {e}")
            raise Exception(f"Failed to retrieve documents of type {document_type}: {str(e)}")

    def _fetch_user_chunks(self, user_id: str, session_id: Optional[str] = None) -> List:
        """Fetch all chunks for a specific user using multiple fallback methods."""
        
        # Use proper filter syntax
        filters = {"user_id": {"$eq": user_id}}
        if session_id:
            filters["session_id"] = {"$eq": session_id}
        
        query_methods = [
            ("query", lambda: self.table.query(filters).to_list()),
            ("sql_query", lambda: self._fetch_user_chunks_sql(user_id, session_id)),
            ("scan_filter", lambda: [c for c in self.table.scan().to_list() 
                                   if c.get('user_id') == user_id and (not session_id or c.get('session_id') == session_id)]),
        ]
        
        for method_name, method_func in query_methods:
            try:
                logger.info(f"Trying fetch method: {method_name} for user: {user_id}")
                chunks = method_func()
                if chunks:
                    logger.info(f"Method '{method_name}' returned {len(chunks)} chunks for user: {user_id}")
                    return chunks
                else:
                    logger.info(f"Method '{method_name}' returned empty result for user: {user_id}")
            except Exception as e:
                logger.warning(f"Method '{method_name}' failed for user {user_id}: {e}")
        
        return []

    def _fetch_user_chunks_sql(self, user_id: str, session_id: Optional[str] = None) -> List:
        """Fetch user chunks using SQL query."""
        if session_id:
            query = "SELECT * FROM user_semantic_embeddings WHERE user_id = :user_id AND session_id = :session_id"
            params = {"user_id": user_id, "session_id": session_id}
        else:
            query = "SELECT * FROM user_semantic_embeddings WHERE user_id = :user_id"
            params = {"user_id": user_id}
        
        return self.tidb_client.query(query, params).to_list()

    def _fetch_all_chunks(self) -> List:
        """Fetch all chunks from all users using multiple fallback methods."""
        
        query_methods = [
            ("scan", lambda: self.table.scan().to_list()),
            ("sql_query", lambda: self.tidb_client.query("SELECT * FROM user_semantic_embeddings").to_list()),
            ("query_all", lambda: self.table.query({}).to_list()),
        ]
        
        for method_name, method_func in query_methods:
            try:
                logger.info(f"Trying fetch method: {method_name} for all documents")
                chunks = method_func()
                if chunks:
                    logger.info(f"Method '{method_name}' returned {len(chunks)} chunks")
                    return chunks
                else:
                    logger.info(f"Method '{method_name}' returned empty result")
            except Exception as e:
                logger.warning(f"Method '{method_name}' failed: {e}")
        
        return []

    def _extract_chunk_data(self, chunk) -> Dict[str, Any]:
        """Extract data from chunk regardless of format (dict or object)."""
        if isinstance(chunk, dict):
            return {
                'document_id': chunk.get('document_id'),
                'document_name': chunk.get('document_name', 'Unknown'),
                'document_type': chunk.get('document_type', 'text'),
                'user_id': chunk.get('user_id'),
                'session_id': chunk.get('session_id'),
                'created_at': chunk.get('created_at'),
                'file_size': chunk.get('file_size'),
            }
        else:
            # Handle object format
            return {
                'document_id': getattr(chunk, 'document_id', None),
                'document_name': getattr(chunk, 'document_name', 'Unknown'),
                'document_type': getattr(chunk, 'document_type', 'text'),
                'user_id': getattr(chunk, 'user_id', None),
                'session_id': getattr(chunk, 'session_id', None),
                'created_at': getattr(chunk, 'created_at', None),
                'file_size': getattr(chunk, 'file_size', None),
            }

    def delete_user_session_documents(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Delete all documents for a user's specific session."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
            
        try:
            # Get chunks count before deletion
            chunks = self._fetch_user_chunks(user_id, session_id)
            chunks_count = len(chunks)
            
            if chunks_count == 0:
                logger.warning(f"No chunks found for session: {session_id} and user: {user_id}")
                return {"chunks_deleted": 0, "user_id": user_id, "session_id": session_id}
            
            # Delete chunks with user and session isolation using proper filter syntax
            filters = {"user_id": {"$eq": user_id}, "session_id": {"$eq": session_id}}
            self.table.delete(filters=filters)
            
            logger.info(f"Deleted {chunks_count} chunks for session: {session_id} and user: {user_id}")
            return {"chunks_deleted": chunks_count, "user_id": user_id, "session_id": session_id}

        except Exception as e:
            logger.error(f"Error deleting session documents from TiDB: {e}")
            raise Exception(f"Failed to delete session documents for user {user_id}, session {session_id}: {str(e)}")

    def delete_all_session_documents(self, session_id: str) -> Dict[str, Any]:
        """Delete all documents for a specific session across all users."""
        
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
            
        try:
            # Get chunks count before deletion
            session_documents = self.get_documents_by_session(session_id)
            total_chunks = sum(doc.get('chunks_count', 0) for doc in session_documents)
            
            if total_chunks == 0:
                logger.warning(f"No chunks found for session: {session_id}")
                return {"chunks_deleted": 0, "session_id": session_id}
            
            # Delete chunks for the session using proper filter syntax
            filters = {"session_id": {"$eq": session_id}}
            self.table.delete(filters=filters)
            
            logger.info(f"Deleted {total_chunks} chunks for session: {session_id}")
            return {"chunks_deleted": total_chunks, "session_id": session_id}

        except Exception as e:
            logger.error(f"Error deleting all session documents from TiDB: {e}")
            raise Exception(f"Failed to delete all session documents for session {session_id}: {str(e)}")

    def delete_all_user_documents(self, user_id: str) -> Dict[str, Any]:
        """Delete all documents for a specific user across all sessions."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
            
        try:
            # Get chunks count before deletion
            user_documents = self.get_documents_by_user(user_id)
            total_chunks = sum(doc.get('chunks_count', 0) for doc in user_documents)
            
            if total_chunks == 0:
                logger.warning(f"No chunks found for user: {user_id}")
                return {"chunks_deleted": 0, "user_id": user_id}
            
            # Delete chunks for the user using proper filter syntax
            filters = {"user_id": {"$eq": user_id}}
            self.table.delete(filters=filters)
            
            logger.info(f"Deleted {total_chunks} chunks for user: {user_id}")
            return {"chunks_deleted": total_chunks, "user_id": user_id}

        except Exception as e:
            logger.error(f"Error deleting all user documents from TiDB: {e}")
            raise Exception(f"Failed to delete all user documents for user {user_id}: {str(e)}")

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user's documents."""
        
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        try:
            user_chunks = self._fetch_user_chunks(user_id)
            
            if not user_chunks:
                return {
                    "user_id": user_id,
                    "total_documents": 0,
                    "total_chunks": 0,
                    "sessions": [],
                    "document_types": {},
                    "total_file_size": 0
                }
            
            # Analyze user's data
            documents = set()
            sessions = set()
            document_types = {}
            total_file_size = 0
            
            for chunk in user_chunks:
                chunk_data = self._extract_chunk_data(chunk)
                
                if chunk_data['document_id']:
                    documents.add(chunk_data['document_id'])
                
                if chunk_data.get('session_id'):
                    sessions.add(chunk_data['session_id'])
                
                doc_type = chunk_data.get('document_type', 'text')
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
                
                if chunk_data.get('file_size'):
                    total_file_size += chunk_data['file_size']
            
            return {
                "user_id": user_id,
                "total_documents": len(documents),
                "total_chunks": len(user_chunks),
                "sessions": list(sessions),
                "document_types": document_types,
                "total_file_size": total_file_size
            }
            
        except Exception as e:
            logger.error(f"Error getting user stats for {user_id}: {e}")
            return {
                "user_id": user_id,
                "error": str(e)
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire document system."""
        
        try:
            all_documents = self.get_all_documents()
            
            if not all_documents:
                return {
                    "total_documents": 0,
                    "total_chunks": 0,
                    "unique_users": 0,
                    "unique_sessions": 0,
                    "document_types": {},
                    "total_file_size": 0,
                    "users_stats": {},
                    "sessions_stats": {}
                }
            
            # Analyze all documents
            unique_users = set()
            unique_sessions = set()
            document_types = {}
            total_file_size = 0
            total_chunks = 0
            users_stats = {}
            sessions_stats = {}
            
            for doc in all_documents:
                # Count chunks
                total_chunks += doc.get('chunks_count', 0)
                
                # Track users
                user_id = doc.get('user_id')
                if user_id:
                    unique_users.add(user_id)
                    if user_id not in users_stats:
                        users_stats[user_id] = {'documents': 0, 'chunks': 0, 'file_size': 0}
                    users_stats[user_id]['documents'] += 1
                    users_stats[user_id]['chunks'] += doc.get('chunks_count', 0)
                    users_stats[user_id]['file_size'] += doc.get('file_size', 0)
                
                # Track sessions
                session_id = doc.get('session_id')
                if session_id:
                    unique_sessions.add(session_id)
                    if session_id not in sessions_stats:
                        sessions_stats[session_id] = {'documents': 0, 'chunks': 0, 'file_size': 0}
                    sessions_stats[session_id]['documents'] += 1
                    sessions_stats[session_id]['chunks'] += doc.get('chunks_count', 0)
                    sessions_stats[session_id]['file_size'] += doc.get('file_size', 0)
                
                # Track document types
                doc_type = doc.get('document_type', 'text')
                document_types[doc_type] = document_types.get(doc_type, 0) + 1
                
                # Track file sizes
                if doc.get('file_size'):
                    total_file_size += doc['file_size']
            
            return {
                "total_documents": len(all_documents),
                "total_chunks": total_chunks,
                "unique_users": len(unique_users),
                "unique_sessions": len(unique_sessions),
                "document_types": document_types,
                "total_file_size": total_file_size,
                "users_stats": users_stats,
                "sessions_stats": sessions_stats,
                "average_chunks_per_document": total_chunks / len(all_documents) if all_documents else 0,
                "average_file_size": total_file_size / len(all_documents) if all_documents else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "total_chunks": 0
            }

    def health_check(self) -> Dict[str, Any]:
        """Check if the user document store is healthy and operational."""
        try:
            health_info = {
                "status": "healthy",
                "table_initialized": self.table is not None,
                "embedding_function": self.embedding_fn is not None,
                "reranker": self.reranker is not None,
            }
            
            if self.table is None:
                return {"status": "unhealthy", "reason": "Table not initialized"}
            
            # Test database connectivity
            try:
                tables = self.tidb_client.list_tables()
                health_info["available_tables"] = tables
                
                if "user_semantic_embeddings" not in tables:
                    return {"status": "unhealthy", "reason": "User document table does not exist"}
                
                # Test query capability
                try:
                    test_chunks = self.table.scan().limit(10).to_list()
                    health_info["sample_chunks"] = len(test_chunks) if test_chunks else 0
                    
                    # Count total documents across all users
                    if test_chunks:
                        unique_users = set()
                        unique_docs = set()
                        for chunk in test_chunks:
                            chunk_data = self._extract_chunk_data(chunk)
                            if chunk_data['user_id']:
                                unique_users.add(chunk_data['user_id'])
                            if chunk_data['document_id']:
                                unique_docs.add(chunk_data['document_id'])
                        health_info["sample_users"] = len(unique_users)
                        health_info["sample_documents"] = len(unique_docs)
                    else:
                        health_info["sample_users"] = 0
                        health_info["sample_documents"] = 0
                        
                except Exception as e:
                    health_info["query_test"] = f"Failed: {str(e)}"
                    return {"status": "degraded", "reason": f"Query test failed: {str(e)}", **health_info}
                
                return {"status": "healthy", **health_info}
                
            except Exception as e:
                return {"status": "unhealthy", "reason": f"Database connectivity issue: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "reason": str(e)}

    def cleanup_orphaned_chunks(self) -> Dict[str, Any]:
        """Clean up chunks that might be orphaned or corrupted."""
        try:
            all_chunks = self._fetch_all_chunks()
            
            if not all_chunks:
                return {"status": "completed", "orphaned_chunks_removed": 0}
            
            orphaned_chunks = []
            valid_chunks = 0
            
            for chunk in all_chunks:
                chunk_data = self._extract_chunk_data(chunk)
                
                # Check for orphaned chunks (missing required fields)
                if (not chunk_data.get('document_id') or 
                    not chunk_data.get('user_id') or 
                    not chunk_data.get('document_name')):
                    orphaned_chunks.append(chunk)
                else:
                    valid_chunks += 1
            
            # Remove orphaned chunks if any found
            orphaned_count = 0
            if orphaned_chunks:
                for chunk in orphaned_chunks:
                    try:
                        chunk_id = chunk.get('id') if isinstance(chunk, dict) else getattr(chunk, 'id', None)
                        if chunk_id:
                            self.table.delete(filters={"id": {"$eq": chunk_id}})
                            orphaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete orphaned chunk: {e}")
            
            logger.info(f"Cleanup completed: {orphaned_count} orphaned chunks removed, {valid_chunks} valid chunks remain")
            
            return {
                "status": "completed",
                "orphaned_chunks_removed": orphaned_count,
                "valid_chunks_remaining": valid_chunks,
                "total_chunks_processed": len(all_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {"status": "failed", "error": str(e)}


# Convenience functions for backward compatibility with user isolation
def search_user_documents(
    user_id: str,
    query: str, 
    document_store: UserDocumentStore, 
    limit: int = 3,
    session_id: Optional[str] = None,
    document_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search documents for a specific user using the document store."""
    return document_store.search(
        user_id=user_id, 
        query=query, 
        limit=limit, 
        session_id=session_id,
        document_type=document_type
    )


def search_all_documents(
    query: str, 
    document_store: UserDocumentStore, 
    limit: int = 3,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    document_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search documents across all users using the document store."""
    return document_store.search_all_documents(
        query=query, 
        limit=limit, 
        user_id=user_id,
        session_id=session_id,
        document_type=document_type
    )


def process_user_text_document(
    user_id: str,
    text: str,
    document_name: str,
    document_store: UserDocumentStore, 
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    document_type: str = "text",
    splitter_type: str = "recursive",
    page_number: Optional[int] = None
) -> Dict[str, Any]:
    """Process and add a text document to the store for a specific user."""
    return document_store.add_document_from_text(
        user_id=user_id,
        text=text,
        document_name=document_name,
        session_id=session_id,
        document_id=document_id,
        document_type=document_type,
        splitter_type=splitter_type,
        page_number=page_number
    )


def process_user_file_document(
    user_id: str,
    file_path: str, 
    document_store: UserDocumentStore, 
    document_name: Optional[str] = None,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process and add text files to the store for a specific user."""
    return document_store.add_document_from_file(
        user_id=user_id,
        file_path=file_path,
        document_name=document_name,
        session_id=session_id,
        document_id=document_id
    )