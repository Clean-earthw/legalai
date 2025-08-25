import os
import uuid
import logging
import re
import dotenv
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field, Column
from pytidb.datatype import TEXT
from pytidb.embeddings import EmbeddingFunction
from pytidb.rerankers import Reranker
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("document_store")


def init_clients():
    """Initialize TiDB client and embedding function."""
    # Check if environment variables are set
    required_vars = ["TIDB_HOST", "TIDB_USERNAME", "TIDB_PASSWORD", "TIDB_DATABASE"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    tidb_client = TiDBClient.connect(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT", 4000)),
        username=os.getenv("TIDB_USERNAME"),
        password=os.getenv("TIDB_PASSWORD"),
        database=os.getenv("TIDB_DATABASE"),
        ensure_db=True,
    )
    
    # Support multiple embedding providers
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "jina").lower()
    
    if embedding_provider == "openai":
        embedding_fn = EmbeddingFunction(
            model_name="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif embedding_provider == "jina":
        embedding_fn = EmbeddingFunction(
            model_name="jina_ai/jina-embeddings-v4",
            api_key=os.getenv("JINA_API_KEY", "jina_73e9e99547164e07a95a36e22fe91eadtNCZpZvdW1MW684HKMG9eJiZi9SY"),
            timeout=20
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
    
    return tidb_client, embedding_fn


# Document schema - will be dynamically created in DocumentStore class
DocumentChunk = None


class DocumentStore:
    """Class to handle document chunks storage and retrieval operations."""
    
    def __init__(self, tidb_client, embedding_fn):
        self.tidb_client = tidb_client
        self.embedding_fn = embedding_fn
        
        # Create the DocumentChunk class dynamically with proper embedding field
        class DocumentChunk(TableModel):
            __tablename__ = "semantic_embeddings"
            __table_args__ = {"extend_existing": True}

            id: int = Field(default=None, primary_key=True)
            text: str = Field(sa_column=Column(TEXT))
            document_id: str
            document_name: str = Field(sa_column=Column(TEXT))
            document_type: str = Field(default="text")  # pdf, txt, docx, etc.
            page_number: Optional[int] = Field(default=None)
            embedding: List[float] = embedding_fn.VectorField(source_field="text")
            section: Optional[str] = Field(sa_column=Column(TEXT), default=None)

        self.DocumentChunk = DocumentChunk
        self.table = tidb_client.create_table(schema=DocumentChunk, if_exists="skip")
        
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
        
        # Initialize reranker if using Jina
        self.reranker = None
        if "jina" in embedding_fn.model_name.lower():
            self.reranker = Reranker(
                model_name="jina_ai/jina-reranker-m0",
            )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text to handle encoding issues."""
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
            
            # Ensure text is a string
            text = str(text)
            
            # Replace problematic characters
            replacements = {
                '\u2018': "'",  # Left single quotation mark
                '\u2019': "'",  # Right single quotation mark
                '\u201c': '"',  # Left double quotation mark
                '\u201d': '"',  # Right double quotation mark
                '\u2013': '-',  # En dash
                '\u2014': '--', # Em dash
                '\u2026': '...', # Horizontal ellipsis
                '\u00a0': ' ',  # Non-breaking space
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
        section_pattern = r'^\s*(\d+(?:\.\d+)?)\s+(.+?)(?=\n|$)'
        match = re.search(section_pattern, text)
        if match:
            return f"{match.group(1)} {match.group(2)}"
        return None
    
    
    def add_document_from_text(
            self, 
            text: str, 
            document_name: str, 
            document_id: Optional[str] = None,
            document_type: str = "text",
            splitter_type: str = "recursive",
            page_number: Optional[int] = None
            ) -> Dict[str, Any]:
        """Add a text document to the store with chunking."""
        if not document_id:
            document_id = str(uuid.uuid4())
            
        # Clean the input text
        try:
            cleaned_text = self._clean_text(text)
            cleaned_document_name = self._clean_text(document_name)
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            raise ValueError(f"Text cleaning failed: {str(e)}")
        
        # Choose splitter
        splitter = self.recursive_splitter if splitter_type == "recursive" else self.character_splitter
        chunks = splitter.split_text(cleaned_text)
        
        if not chunks:
            logger.warning("No chunks created from document")
            return {"document_id": document_id, "chunks_added": 0}
        
        # Create document chunks
        documents = []
        for i, chunk in enumerate(chunks):
            cleaned_chunk = self._clean_text(chunk)
            documents.append(self.DocumentChunk(
                  text=cleaned_chunk,
                  document_id=document_id,
                  document_name=cleaned_document_name,
                  document_type=document_type,
                  page_number=page_number,
                  section=self._identify_section(cleaned_chunk),
            ))
            
            
        try:
            # FIX: Insert documents one by one or use the correct method
            # Option 1: Insert one by one 
            for doc in documents:
                self.table.insert(doc)
        
            # Option 2: If bulk insert is supported, try unpacking
            # self.table.insert(*documents)
            # 
            # Option 3: If the API expects a list, try inserting as dict
            # doc_dicts = [doc.dict() if hasattr(doc, 'dict') else doc.__dict__ for doc in documents]
            # self.table.insert(doc_dicts)
        
            logger.info(f"Successfully added {len(documents)} chunks for document {cleaned_document_name}")
        except Exception as e:
            logger.error(f"Error adding documents to TiDB: {e}")
         # Try alternative insertion method
            try:
                logger.info("Trying alternative insertion method...")
                doc_dicts = []
                for doc in documents:
                    if hasattr(doc, 'dict'):
                        doc_dict = doc.dict()
                    else:
                        doc_dict = doc.__dict__.copy()
                        # Remove any private attributes that start with underscore
                        doc_dict = {k: v for k, v in doc_dict.items() if not k.startswith('_')}
                    doc_dicts.append(doc_dict)
            
                # Try inserting as dictionaries
                for doc_dict in doc_dicts:
                    self.table.insert(doc_dict)
                
                logger.info(f"Successfully added {len(doc_dicts)} chunks using alternative method")
            except Exception as e:
                logger.error(f"Alternative insertion method also failed: {e}")
                raise Exception(f"Both insertion methods failed. Original error: {str(e)}, Alternative error: {str(e)}")
        
        return {"document_id": document_id, "chunks_added": len(documents)}

    def add_document_from_file(
        self, 
        file_path: str, 
        document_name: Optional[str] = None,
        document_id: Optional[str] = None,
        splitter_type: str = "recursive"
    ) -> Dict[str, Any]:
        """Add a document from text file only (TXT, MD)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.txt', '.md']:
            # Use TextLoader for text files
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                documents = loader.load()
                text = "\n".join([doc.page_content for doc in documents])
                
                return self.add_document_from_text(
                    text, 
                    document_name or file_path.stem, 
                    document_id, 
                    "text",
                    splitter_type
                )
            except Exception as e:
                logger.error(f"Error loading text file: {e}")
                raise ValueError(f"Failed to load text file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Use add_document_from_text for pre-processed content.")

    def search(
        self, 
        query: str, 
        limit: int = 3, 
        document_type: Optional[str] = None,
        with_scores: bool = False
    ) -> Union[List[Dict[str, Any]], List[tuple]]:
        """Search for documents using vector similarity search."""
        try:
            cleaned_query = self._clean_text(query)
            
            # Build search query
            search_query = self.table.search(cleaned_query, search_type="vector").limit(limit)
            
            # Add filter for document type if specified
            if document_type:
                search_query = search_query.filter({"document_type": document_type})
            
            # Add reranking if available
            if self.reranker:
                search_query = search_query.rerank(self.reranker, "text")
            
            results = search_query.to_list()
            
            if with_scores:
                # Return results with similarity scores
                scored_results = []
                for result in results:
                    score = result.get('_distance', 0.0)  # Distance from vector search
                    scored_results.append((result, 1 - score))  # Convert distance to similarity
                return scored_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return [] if not with_scores else []

    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 3, 
        document_type: Optional[str] = None
    ) -> List[tuple]:
        """Search for similar documents and return with scores (similar to TiDBVectorStore interface)."""
        return self.search(query, limit=k, document_type=document_type, with_scores=True)

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete all chunks with the given document_id."""
        try:
            # Get chunks count before deletion
            chunks = self.table.query().filter({"document_id": document_id}).to_list()
            chunks_count = len(chunks)

            if chunks_count == 0:
                logger.warning(f"No chunks found with document_id: {document_id}")
                return {"document_id": document_id, "chunks_deleted": 0}
            
            # Delete chunks
            self.table.delete(filters={"document_id": document_id})
            
            logger.info(f"Deleted {chunks_count} chunks with document_id: {document_id}")
            return {"document_id": document_id, "chunks_deleted": chunks_count}

        except Exception as e:
            logger.error(f"Error deleting documents from TiDB: {e}")
            raise

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all unique documents from the store."""
        try:
            # Get all chunks
            all_chunks = self.table.query().to_list()
            
            if not all_chunks:
                logger.info("No documents found in the store")
                return []
            
            # Group by document_id to get unique documents with chunk counts
            documents_map = {}
            for chunk in all_chunks:
                doc_id = chunk['document_id']
                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        'document_id': doc_id,
                        'document_name': chunk['document_name'],
                        'document_type': chunk.get('document_type', 'text'),
                        'chunks_count': 0
                    }
                documents_map[doc_id]['chunks_count'] += 1
            
            documents = list(documents_map.values())
            logger.info(f"Found {len(documents)} unique documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting all documents from TiDB: {e}")
            return []

    def health_check(self) -> Dict[str, Any]:
        """Check if the document store is healthy and operational."""
        try:
            if self.table is None:
                return {"status": "unhealthy", "reason": "Table not available"}
                
            # Test database connectivity
            try:
                tables = self.tidb_client.list_tables()
                
                if "semantic_embeddings" not in tables:
                    return {"status": "unhealthy", "reason": "Table does not exist"}
                    
                return {"status": "healthy", "tables": tables}
            except Exception as e:
                return {"status": "unhealthy", "reason": f"Database connectivity issue: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "reason": str(e)}
        

def search_documents(
    query: str, 
    document_store: DocumentStore, 
    limit: int = 3,
    document_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search documents using the document store."""
    return document_store.search(query=query, limit=limit, document_type=document_type)


def process_text_document(
    text: str,
    document_name: str,
    document_store: DocumentStore, 
    document_id: Optional[str] = None,
    document_type: str = "text",
    splitter_type: str = "recursive",
    page_number: Optional[int] = None
) -> Dict[str, Any]:
    """Process and add a text document to the store."""
    return document_store.add_document_from_text(
        text=text,
        document_name=document_name,
        document_id=document_id,
        document_type=document_type,
        splitter_type=splitter_type,
        page_number=page_number
    )


def process_file_document(
    file_path: str, 
    document_store: DocumentStore, 
    document_name: Optional[str] = None,
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process and add text files to the store."""
    return document_store.add_document_from_file(
        file_path=file_path,
        document_name=document_name,
        document_id=document_id
    )