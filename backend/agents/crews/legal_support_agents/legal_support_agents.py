import os
import yaml
import logging
import sys
import json
from pathlib import Path
from typing import Type, Optional, Dict, Any, List
from enum import Enum
from openai import OpenAI, AsyncOpenAI
import instructor

from pydantic import BaseModel, Field
from agents.rag.document_store import DocumentStore, init_clients, search_documents

# Configure minimal logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('legal_support_agents')

class AgentName(str, Enum):
    EMPLOYMENT = "Employment Expert"
    COMPLIANCE = "Compliance Specialist"
    EQUITY = "Equity Management Expert"

class RoutingDecision(BaseModel):
    agent_name: AgentName
    confidence: float = Field(description="Confidence score between 0-1")
    reasoning: Optional[str] = Field(description="Brief explanation for routing decision")

class Answer(BaseModel):
    content: str
    sources_used: Optional[List[str]] = Field(description="List of document sources referenced")
    confidence: Optional[float] = Field(description="Confidence in the answer")

class DocumentSearchResult(BaseModel):
    query: str
    results_count: int
    documents: List[Dict[str, Any]]

class LegalSupportAgents:
    BASE_DIR = Path(__file__).parent

    def __init__(self, debug_enabled: bool = False):
        self.debug_enabled = debug_enabled

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.ai_base_url = os.getenv("AI_BASE_URL")
        self.gen_ai_model = os.getenv("GEN_AI_MODEL")

        # Load the agents configuration files
        self.agents_config_path = os.path.join(self.BASE_DIR, "configs", "agents.yaml")
        self.tasks_config_path = os.path.join(self.BASE_DIR, "configs", "tasks.yaml")

        try:
            with open(self.agents_config_path, 'r') as f:
                self.agents_configs = yaml.safe_load(f)
            with open(self.tasks_config_path, 'r') as f:
                self.tasks_configs = yaml.safe_load(f)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file is not found: {e.filename}") from e
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing yaml configuration: {e}") from e

        # Set up AsyncOpenAI client and patch with Instructor
        ai_client = AsyncOpenAI(
            api_key=self.gemini_api_key,
            base_url=self.ai_base_url
        )
        
        # Patch the client with instructor
        self.client = instructor.apatch(ai_client)

        # Initialize DocumentStore - now synchronous
        self.document_store = None
        self.rag_initialized = False

    def initialize_rag(self) -> bool:
        """Initialize the RAG document store synchronously."""
        if self.rag_initialized:
            return True
            
        try:
            # Initialize TiDB client and embedding function
            tidb_client, embedding_fn = init_clients()
            
            # Create DocumentStore instance
            self.document_store = DocumentStore(tidb_client, embedding_fn)
            
            # Test the connection
            health_status = self.document_store.health_check()
            if health_status["status"] != "healthy":
                logger.error(f"Document store health check failed: {health_status}")
                return False
                
            self.rag_initialized = True
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_initialized = False
            return False

    async def process_query(self, query: str) -> str:
        """
        Process a user query by routing it and generating answer
        Args:
            query: user text input
        """
        try:
            orchestrator_config = self.agents_configs["orchestrator"]
            employment_config = self.agents_configs["employment_expert"]
            compliance_config = self.agents_configs["compliance_specialist"]
            equity_config = self.agents_configs["equity_management_expert"]
            
            # Construct prompt for routing the query
            routing_prompt = self.tasks_configs["route_request"]["description"].format(
                query=query,
                agent_role=orchestrator_config["role"],
                agent_goal=orchestrator_config["goal"],
                agent_backstory=orchestrator_config["backstory"],
                routing_guidelines="\n".join([f"- {item}" for item in orchestrator_config.get("routing_guidelines", [])]),
                tone=orchestrator_config.get("tone")
            )

            # Log request inspection details if debug is enabled
            log_request_inspection(model_type=RoutingDecision, prompt=routing_prompt, agent_name="ROUTING", enabled=self.debug_enabled)

            routing_decision = await self.client.chat.completions.create(
                model=self.gen_ai_model,
                messages=[{"role": "user", "content": routing_prompt}],
                response_format=RoutingDecision,
            )

            # Define agent handlers with their corresponding configs
            agent_handlers = {
                AgentName.EMPLOYMENT: lambda q: self._handle_employment_query(q, employment_config),
                AgentName.COMPLIANCE: lambda q: self._handle_compliance_query(q, compliance_config),
                AgentName.EQUITY: lambda q: self._handle_equity_query(q, equity_config)
            }

            # Use the appropriate handler or return a default message
            handler = agent_handlers.get(routing_decision.agent_name)
            if handler:
                result = await handler(query)
                
                # Add routing information if debug is enabled
                if self.debug_enabled and routing_decision.reasoning:
                    result += f"\n\n*Routing: {routing_decision.reasoning} (Confidence: {routing_decision.confidence:.2f})*"
                
                return result
            
            return "**[Support Request Orchestrator]** I'm sorry, but I cannot answer that question."

        except Exception as e:
            logger.error(f"Unexpected error processing query: {e}", exc_info=True)
            return "An error occurred while processing your query. Please try again."
        
    async def _handle_employment_query(self, query: str, employment_config: dict) -> str:
        """ Handle queries related to employment and stock options."""
        try:
            relevant_context = self.get_relevant_context(query)

            employment_prompt = self.tasks_configs["answer_employment_question"]["description"].format(
                query=query, 
                relevant_context=relevant_context["context"],
                agent_role=employment_config["role"],
                agent_goal=employment_config["goal"],
                agent_backstory=employment_config["backstory"],
                expertise_areas="\n".join([f"- {item}" for item in employment_config.get("expertise_areas", [])]),
                response_guidelines=employment_config.get("response_guidelines"),
                tone=employment_config.get("tone")
            )
            log_request_inspection(model_type=Answer, prompt=employment_prompt, agent_name="EMPLOYMENT", enabled=self.debug_enabled)

            answer = await self.client.chat.completions.create(
                 model=self.gen_ai_model,
                 messages=[{"role": "user", "content": employment_prompt}],
                 response_format=Answer,
            )
            
            # Format response with source information
            response = f"**[Employment Expert]** {answer.content}"
            if answer.sources_used and len(relevant_context["sources"]) > 0:
                response += f"\n\n*Sources: {', '.join(relevant_context['sources'])}*"
                
            return response
            
        except Exception as e:
            logger.error(f"Error in employment query handler: {e}")
            return "**[Employment Expert]** I'm sorry, I encountered an error processing your employment question. Please try again."
    
    async def _handle_compliance_query(self, query: str, compliance_config: dict) -> str:
        """ Handle queries related to compliance and regulatory requirements."""
        try:
            # Compliance queries might not always need RAG context, but we'll try to get it
            relevant_context = self.get_relevant_context(query)
            
            compliance_prompt = self.tasks_configs["answer_compliance_question"]["description"].format(
                query=query,
                relevant_context=relevant_context["context"] if relevant_context["context"] != "No relevant documents found in the knowledge base." else "",
                agent_role=compliance_config["role"],
                agent_goal=compliance_config["goal"],
                agent_backstory=compliance_config["backstory"],
                response_guidelines=compliance_config.get("response_guidelines"),
                tone=compliance_config.get("tone")
            )
            
            log_request_inspection(model_type=Answer, prompt=compliance_prompt, agent_name="COMPLIANCE", enabled=self.debug_enabled)
            
            answer = await self.client.chat.completions.create(
                model=self.gen_ai_model,
                messages=[{"role": "user", "content": compliance_prompt}],
                response_format=Answer,
            )
            
            # Format response with source information
            response = f"**[Compliance Specialist]** {answer.content}"
            if answer.sources_used and len(relevant_context["sources"]) > 0:
                response += f"\n\n*Sources: {', '.join(relevant_context['sources'])}*"
                
            return response
            
        except Exception as e:
            logger.error(f"Error in compliance query handler: {e}")
            return "**[Compliance Specialist]** I'm sorry, I encountered an error processing your compliance question. Please try again."
    
    async def _handle_equity_query(self, query: str, equity_config: dict) -> str:
        """
        Handle queries related to equity management and company structure.
        
        Args:
            query: The user's query text
            equity_config: The configuration for the equity management expert agent
        """
        try:
            relevant_context = self.get_relevant_context(query)
            
            equity_prompt = self.tasks_configs["answer_equity_question"]["description"].format(
                query=query,
                relevant_context=relevant_context["context"],
                agent_role=equity_config["role"],
                agent_goal=equity_config["goal"],
                agent_backstory=equity_config["backstory"],
                expertise_areas="\n".join([f"- {item}" for item in equity_config.get("expertise_areas", [])]),
                response_guidelines=equity_config.get("response_guidelines"),
                tone=equity_config.get("tone")
            )
            
            log_request_inspection(model_type=Answer, prompt=equity_prompt, agent_name="EQUITY", enabled=self.debug_enabled)
            
            answer = await self.client.chat.completions.create(
                model=self.gen_ai_model,
                messages=[{"role": "user", "content": equity_prompt}],
                response_format=Answer,
            )
            
            # Format response with source information
            response = f"**[Equity Management Expert]** {answer.content}"
            if answer.sources_used and len(relevant_context["sources"]) > 0:
                response += f"\n\n*Sources: {', '.join(relevant_context['sources'])}*"
                
            return response
            
        except Exception as e:
            logger.error(f"Error in equity query handler: {e}")
            return "**[Equity Management Expert]** I'm sorry, I encountered an error processing your equity question. Please try again."

    def search_documents(self, query: str, limit: int = 3, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents in the document store."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    return []
            
            results = self.document_store.search(query, limit=limit, document_type=document_type)
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []  # Return empty results instead of raising
    
    def get_relevant_context(self, query: str, limit: int = 3) -> Dict[str, Any]:
        """Retrieve and format relevant document context for the query."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    return {
                        "context": "Document search is currently unavailable.",
                        "sources": []
                    }
            
            results = self.search_documents(query, limit=limit)

            if not results or len(results) == 0:
                return {
                    "context": "No relevant documents found in the knowledge base.",
                    "sources": []
                }
                
            context = "Here is relevant information from our documents:\n\n"
            sources = []
            
            for i, result in enumerate(results):
                document_name = result.get('document_name', 'Unknown')
                sources.append(document_name)
                
                context += f"Document {i+1}: {document_name}"
                if result.get('document_type'):
                    context += f" (Type: {result['document_type']})"
                if result.get('page_number'):
                    context += f" (Page: {result['page_number']})"
                context += "\n"
                
                if result.get('section'):
                    context += f"Section: {result['section']}\n"
                context += f"Content: {result.get('text', '')}\n\n"
                
            return {
                "context": context,
                "sources": list(set(sources))  # Remove duplicates
            }
            
        except Exception as e:
            logger.error(f"Error retrieving document context: {e}")
            return {
                "context": "Could not retrieve relevant documents due to a system error.",
                "sources": []
            }

    def add_document_from_file(self, file_path: str, document_name: Optional[str] = None, 
                              document_id: Optional[str] = None, splitter_type: str = "recursive") -> Dict[str, Any]:
        """Add a document from file to the knowledge base."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    raise RuntimeError("Failed to initialize RAG system")
            
            return self.document_store.add_document_from_file(
                file_path=file_path,
                document_name=document_name,
                document_id=document_id,
                splitter_type=splitter_type
            )
        except Exception as e:
            logger.error(f"Error adding document from file: {e}")
            raise
        
    def add_document_from_text(self, 
                          text: str, 
                          document_name: str, 
                          document_id: Optional[str] = None,
                          document_type: str = "text",  # Moved to correct position
                          splitter_type: str = "recursive") -> Dict[str, Any]:
        """Add a text document to the knowledge base."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    raise RuntimeError("Failed to initialize RAG system")
                
            return self.document_store.add_document_from_text(
                 text=text,
                 document_name=document_name,
                 document_id=document_id,
                 document_type=document_type,
                 splitter_type=splitter_type
            )
        except Exception as e:
            logger.error(f"Error adding text document: {e}")
            raise

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document from the knowledge base."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    raise RuntimeError("Failed to initialize RAG system")
            
            return self.document_store.delete_document(document_id)
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from the knowledge base."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    return []
            
            return self.document_store.get_all_documents()
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    async def search_documents_structured(self, query: str, limit: int = 3, 
                                         document_type: Optional[str] = None) -> DocumentSearchResult:
        """Search documents and return structured results."""
        try:
            results = self.search_documents(query, limit=limit, document_type=document_type)
            
            return DocumentSearchResult(
                query=query,
                results_count=len(results),
                documents=results
            )
        except Exception as e:
            logger.error(f"Error in structured document search: {e}")
            return DocumentSearchResult(
                query=query,
                results_count=0,
                documents=[]
            )

    def similarity_search_with_score(self, query: str, k: int = 3, 
                                   document_type: Optional[str] = None) -> List[tuple]:
        """Search for similar documents and return with scores."""
        try:
            if not self.rag_initialized:
                if not self.initialize_rag():
                    return []
            
            return self.document_store.similarity_search_with_score(
                query=query, k=k, document_type=document_type
            )
        except Exception as e:
            logger.error(f"Error in similarity search with scores: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the legal support agents system."""
        try:
            health_status = {
                "status": "healthy",
                "components": {
                    "agents_config": "loaded" if hasattr(self, 'agents_configs') else "not_loaded",
                    "tasks_config": "loaded" if hasattr(self, 'tasks_configs') else "not_loaded",
                    "ai_client": "connected" if self.client else "disconnected",
                    "rag_system": "initialized" if self.rag_initialized else "not_initialized"
                }
            }
            
            # Check document store health if initialized
            if self.rag_initialized and self.document_store:
                doc_health = self.document_store.health_check()
                health_status["components"]["document_store"] = doc_health["status"]
                if doc_health["status"] != "healthy":
                    health_status["document_store_details"] = doc_health
            else:
                health_status["components"]["document_store"] = "not_initialized"
            
            # Try to initialize RAG if not already done
            if not self.rag_initialized:
                if self.initialize_rag():
                    health_status["components"]["rag_system"] = "initialized"
                    doc_health = self.document_store.health_check()
                    health_status["components"]["document_store"] = doc_health["status"]
            
            # Check if all components are healthy
            unhealthy_components = [k for k, v in health_status["components"].items() 
                                  if v not in ["loaded", "connected", "initialized", "healthy"]]
            
            if unhealthy_components:
                health_status["status"] = "degraded"
                health_status["issues"] = unhealthy_components
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Function to log request inspection details
def log_request_inspection(model_type: Type[BaseModel], prompt: str, agent_name: str = "INSTRUCTOR", enabled: bool = True):
    """
    Log the request inspection details for an agent call.
    
    Args:
        model_type: The Pydantic model type being used for the response
        prompt: The prompt text being sent to the agent
        agent_name: Name of the agent for logging purposes
        enabled: Whether logging is enabled
    """
    if not enabled:
        return
        
    # Get the Pydantic schema for the model
    schema = model_type.model_json_schema()
    
    print(f"\n\n==== {agent_name} REQUEST INSPECTION ====")
    print("Messages:", json.dumps([{"role": "user", "content": prompt}], indent=2))
    print("Pydantic Schema:", json.dumps(schema, indent=2))
    print("=================================================\n\n")


# Convenience functions for backward compatibility and ease of use
def create_legal_support_system(debug_enabled: bool = False) -> LegalSupportAgents:
    """Create and initialize a legal support agents system."""
    return LegalSupportAgents(debug_enabled=debug_enabled)

def search_knowledge_base(agents_system: LegalSupportAgents, query: str, 
                         limit: int = 3, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search the knowledge base for relevant documents."""
    return agents_system.search_documents(query, limit=limit, document_type=document_type)

def add_document_to_knowledge_base(agents_system: LegalSupportAgents, file_path: str, 
                                  document_name: Optional[str] = None) -> Dict[str, Any]:
    """Add a document to the knowledge base."""
    return agents_system.add_document_from_file(file_path, document_name)