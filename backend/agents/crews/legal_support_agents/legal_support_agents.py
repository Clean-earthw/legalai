import os
import yaml
import logging
import sys
import json
from pathlib import Path
from typing import Type, Optional, Dict, Any, List
from enum import Enum
from openai import OpenAI
import instructor

from pydantic import BaseModel, Field
from agents.rag.document_store import UserDocumentStore, init_clients

# Configure minimal logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('user_legal_support_agents')


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
    user_id: str
    session_id: Optional[str] = None


class UserLegalSupportAgents:
    """Legal support agents system with user-specific document isolation."""
    
    BASE_DIR = Path(__file__).parent

    def __init__(self, debug_enabled: bool = False):
        """Initialize the legal support agents system."""
        self.debug_enabled = debug_enabled
        
        # Get API configuration
        self._validate_environment()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.ai_base_url = os.getenv("AI_BASE_URL")
        self.gen_ai_model = os.getenv("GEN_AI_MODEL")

        # Load configurations
        self._load_configurations()
        
        # Initialize AI client
        self._initialize_ai_client()
        
        # Initialize RAG system
        self.document_store = None
        self.rag_initialized = False

    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        required_vars = ["GEMINI_API_KEY", "GEN_AI_MODEL"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    def _load_configurations(self) -> None:
        """Load agent and task configurations from YAML files."""
        self.agents_config_path = self.BASE_DIR / "config" / "agents.yaml"
        self.tasks_config_path = self.BASE_DIR / "config" / "tasks.yaml"

        try:
            with open(self.agents_config_path, 'r') as f:
                self.agents_configs = yaml.safe_load(f)
            with open(self.tasks_config_path, 'r') as f:
                self.tasks_configs = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {e.filename}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")

    def _initialize_ai_client(self) -> None:
        """Initialize and patch the OpenAI client with Instructor."""
        ai_client = OpenAI(
            api_key=self.gemini_api_key,
            base_url=self.ai_base_url
        )
        self.client = instructor.from_openai(ai_client)

    def initialize_rag(self) -> bool:
        """Initialize the RAG document store."""
        if self.rag_initialized:
            return True
            
        try:
            tidb_client, embedding_fn = init_clients()
            self.document_store = UserDocumentStore(tidb_client, embedding_fn)
            
            # Verify health
            health_status = self.document_store.health_check()
            if health_status["status"] not in ["healthy", "degraded"]:
                logger.error(f"Document store health check failed: {health_status}")
                return False
                
            self.rag_initialized = True
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_initialized = False
            return False

    def process_query(self, user_id: str, query: str, session_id: Optional[str] = None) -> str:
        """
        Process a user query by routing it to the appropriate agent.
        
        Args:
            user_id: User identifier for document isolation
            query: User text input
            session_id: Optional session identifier
            
        Returns:
            str: Formatted response from the appropriate agent
        """
        if not user_id or not user_id.strip():
            return "**[Error]** User ID is required to process queries."

        try:
            # Route the query
            routing_decision = self._route_query(query)
            
            # Handle the query with the appropriate agent
            agent_handlers = {
                AgentName.EMPLOYMENT: self._handle_employment_query,
                AgentName.COMPLIANCE: self._handle_compliance_query,
                AgentName.EQUITY: self._handle_equity_query
            }

            handler = agent_handlers.get(routing_decision.agent_name)
            if not handler:
                return "**[Support Request Orchestrator]** I'm sorry, but I cannot answer that question."
            
            result = handler(user_id, query, session_id)
            
            # Add routing information if debug is enabled
            if self.debug_enabled and routing_decision.reasoning:
                result += f"\n\n*Routing: {routing_decision.reasoning} (Confidence: {routing_decision.confidence:.2f})*"
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error processing query for user {user_id}: {e}", exc_info=True)
            return "An error occurred while processing your query. Please try again."

    def _route_query(self, query: str) -> RoutingDecision:
        """Route the query to the appropriate agent."""
        orchestrator_config = self.agents_configs["orchestrator"]
        
        routing_prompt = self.tasks_configs["route_request"]["description"].format(
            query=query,
            agent_role=orchestrator_config["role"],
            agent_goal=orchestrator_config["goal"],
            agent_backstory=orchestrator_config["backstory"],
            routing_guidelines="\n".join([f"- {item}" for item in orchestrator_config.get("routing_guidelines", [])]),
            tone=orchestrator_config.get("tone", "professional")
        )

        self._log_request_inspection(RoutingDecision, routing_prompt, "ROUTING")

        return self.client.chat.completions.create(
            model=self.gen_ai_model,
            messages=[{"role": "user", "content": routing_prompt}],
            response_model=RoutingDecision,
        )

    def _handle_employment_query(self, user_id: str, query: str, session_id: Optional[str] = None) -> str:
        """Handle employment-related queries."""
        try:
            employment_config = self.agents_configs["employment_expert"]
            relevant_context = self.get_user_relevant_context(user_id, query, session_id)

            prompt = self.tasks_configs["answer_employment_question"]["description"].format(
                query=query, 
                relevant_context=relevant_context["context"],
                agent_role=employment_config["role"],
                agent_goal=employment_config["goal"],
                agent_backstory=employment_config["backstory"],
                expertise_areas="\n".join([f"- {item}" for item in employment_config.get("expertise_areas", [])]),
                response_guidelines=employment_config.get("response_guidelines", ""),
                tone=employment_config.get("tone", "professional")
            )
            
            self._log_request_inspection(Answer, prompt, "EMPLOYMENT")

            answer = self.client.chat.completions.create(
                 model=self.gen_ai_model,
                 messages=[{"role": "user", "content": prompt}],
                 response_model=Answer,
            )
            
            return self._format_response("Employment Expert", answer, relevant_context)
            
        except Exception as e:
            logger.error(f"Error in employment query handler for user {user_id}: {e}")
            return "**[Employment Expert]** I'm sorry, I encountered an error processing your employment question. Please try again."

    def _handle_compliance_query(self, user_id: str, query: str, session_id: Optional[str] = None) -> str:
        """Handle compliance-related queries."""
        try:
            compliance_config = self.agents_configs["compliance_specialist"]
            relevant_context = self.get_user_relevant_context(user_id, query, session_id)
            
            prompt = self.tasks_configs["answer_compliance_question"]["description"].format(
                query=query,
                relevant_context=relevant_context["context"] if relevant_context["context"] != "No relevant documents found in your knowledge base." else "",
                agent_role=compliance_config["role"],
                agent_goal=compliance_config["goal"],
                agent_backstory=compliance_config["backstory"],
                response_guidelines=compliance_config.get("response_guidelines", ""),
                tone=compliance_config.get("tone", "professional")
            )
            
            self._log_request_inspection(Answer, prompt, "COMPLIANCE")
            
            answer = self.client.chat.completions.create(
                model=self.gen_ai_model,
                messages=[{"role": "user", "content": prompt}],
                response_model=Answer,
            )
            
            return self._format_response("Compliance Specialist", answer, relevant_context)
            
        except Exception as e:
            logger.error(f"Error in compliance query handler for user {user_id}: {e}")
            return "**[Compliance Specialist]** I'm sorry, I encountered an error processing your compliance question. Please try again."

    def _handle_equity_query(self, user_id: str, query: str, session_id: Optional[str] = None) -> str:
        """Handle equity management queries."""
        try:
            equity_config = self.agents_configs["equity_management_expert"]
            relevant_context = self.get_user_relevant_context(user_id, query, session_id)
            
            prompt = self.tasks_configs["answer_equity_question"]["description"].format(
                query=query,
                relevant_context=relevant_context["context"],
                agent_role=equity_config["role"],
                agent_goal=equity_config["goal"],
                agent_backstory=equity_config["backstory"],
                expertise_areas="\n".join([f"- {item}" for item in equity_config.get("expertise_areas", [])]),
                response_guidelines=equity_config.get("response_guidelines", ""),
                tone=equity_config.get("tone", "professional")
            )
            
            self._log_request_inspection(Answer, prompt, "EQUITY")
            
            answer = self.client.chat.completions.create(
                model=self.gen_ai_model,
                messages=[{"role": "user", "content": prompt}],
                response_model=Answer,
            )
            
            return self._format_response("Equity Management Expert", answer, relevant_context)
            
        except Exception as e:
            logger.error(f"Error in equity query handler for user {user_id}: {e}")
            return "**[Equity Management Expert]** I'm sorry, I encountered an error processing your equity question. Please try again."

    def _format_response(self, agent_name: str, answer: Answer, relevant_context: Dict[str, Any]) -> str:
        """Format the agent response with source information."""
        response = f"**[{agent_name}]** {answer.content}"
        
        if answer.sources_used and len(relevant_context["sources"]) > 0:
            response += f"\n\n*Sources from your documents: {', '.join(relevant_context['sources'])}*"
        elif len(relevant_context["sources"]) == 0:
            response += f"\n\n*Note: This response is based on general knowledge. Upload relevant documents for personalized advice.*"
            
        return response

    def _log_request_inspection(self, model_type: Type[BaseModel], prompt: str, agent_name: str) -> None:
        """Log request inspection details if debug is enabled."""
        if not self.debug_enabled:
            return
            
        schema = model_type.model_json_schema()
        print(f"\n\n==== {agent_name} REQUEST INSPECTION ====")
        print("Messages:", json.dumps([{"role": "user", "content": prompt}], indent=2))
        print("Pydantic Schema:", json.dumps(schema, indent=2))
        print("=================================================\n\n")

    # Document Management Methods
    def get_user_relevant_context(self, user_id: str, query: str, session_id: Optional[str] = None, limit: int = 3) -> Dict[str, Any]:
        """Retrieve and format relevant document context for the user's query."""
        if not user_id or not user_id.strip():
            return {"context": "User identification required for document search.", "sources": []}
        
        try:
            if not self.rag_initialized and not self.initialize_rag():
                return {"context": "Document search is currently unavailable.", "sources": []}
            
            results = self.search_user_documents(user_id, query, limit=limit, session_id=session_id)

            if not results:
                return {"context": "No relevant documents found in your knowledge base.", "sources": []}
                
            context = "Here is relevant information from your uploaded documents:\n\n"
            sources = []
            
            for i, result in enumerate(results):
                document_name = result.get('document_name', 'Unknown')
                sources.append(document_name)
                
                context += f"Document {i+1}: {document_name}"
                if result.get('document_type'):
                    context += f" (Type: {result['document_type']})"
                if result.get('page_number'):
                    context += f" (Page: {result['page_number']})"
                if result.get('session_id'):
                    context += f" (Session: {result['session_id']})"
                context += "\n"
                
                if result.get('section'):
                    context += f"Section: {result['section']}\n"
                context += f"Content: {result.get('text', '')}\n\n"
                
            return {"context": context, "sources": list(set(sources))}
            
        except Exception as e:
            logger.error(f"Error retrieving document context for user {user_id}: {e}")
            return {"context": "Could not retrieve relevant documents due to a system error.", "sources": []}

    def search_user_documents(self, user_id: str, query: str, limit: int = 3, 
                             session_id: Optional[str] = None, document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents in the user's document store."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        try:
            if not self.rag_initialized and not self.initialize_rag():
                return []
            
            return self.document_store.search(user_id, query, limit=limit, 
                                            session_id=session_id, document_type=document_type)
        except Exception as e:
            logger.error(f"Error searching documents for user {user_id}: {e}")
            return []

    def add_user_document_from_file(self, user_id: str, file_path: str, document_name: Optional[str] = None, 
                                   session_id: Optional[str] = None, document_id: Optional[str] = None, 
                                   splitter_type: str = "recursive") -> Dict[str, Any]:
        """Add a document from file to the user's knowledge base."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            raise RuntimeError("Failed to initialize RAG system")
        
        return self.document_store.add_document_from_file(
            user_id=user_id, file_path=file_path, document_name=document_name,
            session_id=session_id, document_id=document_id, splitter_type=splitter_type
        )
        
    def add_user_document_from_text(self, user_id: str, text: str, document_name: str, 
                                   session_id: Optional[str] = None, document_id: Optional[str] = None,
                                   document_type: str = "text", splitter_type: str = "recursive",
                                   page_number: Optional[int] = None) -> Dict[str, Any]:
        """Add a text document to the user's knowledge base."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            raise RuntimeError("Failed to initialize RAG system")
                
        return self.document_store.add_document_from_text(
            user_id=user_id, text=text, document_name=document_name,
            session_id=session_id, document_id=document_id, document_type=document_type,
            splitter_type=splitter_type, page_number=page_number
        )

    def delete_user_document(self, user_id: str, document_id: str) -> Dict[str, Any]:
        """Delete a document from the user's knowledge base."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            raise RuntimeError("Failed to initialize RAG system")
        
        return self.document_store.delete_document(user_id, document_id)

    def delete_user_session_documents(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Delete all documents for a user's specific session."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        if not session_id or not session_id.strip():
            raise ValueError("Session ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            raise RuntimeError("Failed to initialize RAG system")
        
        return self.document_store.delete_user_session_documents(user_id, session_id)

    def delete_all_user_documents(self, user_id: str) -> Dict[str, Any]:
        """Delete all documents for a specific user across all sessions."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            raise RuntimeError("Failed to initialize RAG system")
        
        return self.document_store.delete_all_user_documents(user_id)

    def get_user_documents(self, user_id: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all documents from the user's knowledge base."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            return []
        
        if session_id:
            return self.document_store.get_documents_by_session(session_id)
        else:
            return self.document_store.get_documents_by_user(user_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from all users and sessions."""
        if not self.rag_initialized and not self.initialize_rag():
            return []
        
        return self.document_store.get_all_documents()

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user's documents."""
        if not user_id or not user_id.strip():
            raise ValueError("User ID cannot be empty")
        
        if not self.rag_initialized and not self.initialize_rag():
            return {"user_id": user_id, "error": "RAG system not initialized"}
        
        return self.document_store.get_user_stats(user_id)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the entire document system."""
        if not self.rag_initialized and not self.initialize_rag():
            return {"error": "RAG system not initialized"}
        
        return self.document_store.get_system_stats()

    def search_user_documents_structured(self, user_id: str, query: str, limit: int = 3, 
                                        session_id: Optional[str] = None,
                                        document_type: Optional[str] = None) -> DocumentSearchResult:
        """Search documents and return structured results for a specific user."""
        if not user_id or not user_id.strip():
            return DocumentSearchResult(
                query=query, results_count=0, documents=[],
                user_id="", session_id=session_id
            )
        
        try:
            results = self.search_user_documents(
                user_id, query, limit=limit, session_id=session_id, document_type=document_type
            )
            
            return DocumentSearchResult(
                query=query, results_count=len(results), documents=results,
                user_id=user_id, session_id=session_id
            )
        except Exception as e:
            logger.error(f"Error in structured document search for user {user_id}: {e}")
            return DocumentSearchResult(
                query=query, results_count=0, documents=[],
                user_id=user_id, session_id=session_id
            )

    def similarity_search_with_score(self, user_id: str, query: str, k: int = 3, 
                                    session_id: Optional[str] = None,
                                    document_type: Optional[str] = None) -> List[tuple]:
        """Search for similar documents and return with scores for a specific user."""
        if not user_id or not user_id.strip():
            return []
        
        if not self.rag_initialized and not self.initialize_rag():
            return []
        
        return self.document_store.similarity_search_with_score(
            user_id=user_id, query=query, k=k, session_id=session_id, document_type=document_type
        )

    def search_all_documents(self, query: str, limit: int = 3, 
                            user_id: Optional[str] = None, session_id: Optional[str] = None,
                            document_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documents across all users."""
        if not self.rag_initialized and not self.initialize_rag():
            return []
        
        return self.document_store.search_all_documents(
            query=query, limit=limit, user_id=user_id,
            session_id=session_id, document_type=document_type
        )

    def get_documents_by_type(self, document_type: str) -> List[Dict[str, Any]]:
        """Retrieve all documents of a specific type."""
        if not self.rag_initialized and not self.initialize_rag():
            return []
        
        return self.document_store.get_documents_by_type(document_type)

    def cleanup_orphaned_chunks(self) -> Dict[str, Any]:
        """Clean up chunks that might be orphaned or corrupted."""
        if not self.rag_initialized and not self.initialize_rag():
            return {"status": "failed", "error": "RAG system not initialized"}
        
        return self.document_store.cleanup_orphaned_chunks()

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the user legal support agents system."""
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
                if doc_health["status"] not in ["healthy", "degraded"]:
                    health_status["document_store_details"] = doc_health
            else:
                health_status["components"]["document_store"] = "not_initialized"
            
            # Try to initialize RAG if not already done
            if not self.rag_initialized:
                try:
                    if self.initialize_rag():
                        health_status["components"]["rag_system"] = "initialized"
                        if self.document_store:
                            doc_health = self.document_store.health_check()
                            health_status["components"]["document_store"] = doc_health["status"]
                except Exception as e:
                    logger.warning(f"RAG initialization failed during health check: {e}")
            
            # Check if all components are healthy
            unhealthy_components = [k for k, v in health_status["components"].items() 
                                  if v not in ["loaded", "connected", "initialized", "healthy", "degraded"]]
            
            if unhealthy_components:
                health_status["status"] = "degraded"
                health_status["issues"] = unhealthy_components
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}


# Convenience functions for ease of use
def create_user_legal_support_system(debug_enabled: bool = False) -> UserLegalSupportAgents:
    """Create and initialize a user legal support agents system."""
    return UserLegalSupportAgents(debug_enabled=debug_enabled)


def process_user_legal_query(agents_system: UserLegalSupportAgents, user_id: str, 
                           query: str, session_id: Optional[str] = None) -> str:
    """Process a legal query with user-specific context."""
    return agents_system.process_query(user_id, query, session_id)


def search_user_knowledge_base(agents_system: UserLegalSupportAgents, user_id: str, query: str, 
                              limit: int = 3, session_id: Optional[str] = None,
                              document_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search the user's knowledge base for relevant documents."""
    return agents_system.search_user_documents(user_id, query, limit=limit, 
                                             session_id=session_id, document_type=document_type)


def add_user_document_to_knowledge_base(agents_system: UserLegalSupportAgents, user_id: str, 
                                       file_path: str, document_name: Optional[str] = None,
                                       session_id: Optional[str] = None, 
                                       document_id: Optional[str] = None,
                                       splitter_type: str = "recursive") -> Dict[str, Any]:
    """Add a document to the user's knowledge base."""
    return agents_system.add_user_document_from_file(
        user_id, file_path, document_name, session_id, document_id, splitter_type
    )


def add_user_text_to_knowledge_base(agents_system: UserLegalSupportAgents, user_id: str,
                                   text: str, document_name: str, 
                                   session_id: Optional[str] = None,
                                   document_id: Optional[str] = None,
                                   document_type: str = "text",
                                   splitter_type: str = "recursive",
                                   page_number: Optional[int] = None) -> Dict[str, Any]:
    """Add text content to the user's knowledge base."""
    return agents_system.add_user_document_from_text(
        user_id=user_id, text=text, document_name=document_name,
        session_id=session_id, document_id=document_id, document_type=document_type,
        splitter_type=splitter_type, page_number=page_number
    )


def get_user_document_stats(agents_system: UserLegalSupportAgents, user_id: str) -> Dict[str, Any]:
    """Get statistics for a user's documents."""
    return agents_system.get_user_stats(user_id)


def get_system_document_stats(agents_system: UserLegalSupportAgents) -> Dict[str, Any]:
    """Get comprehensive statistics for the entire document system."""
    return agents_system.get_system_stats()


def delete_user_document_from_knowledge_base(agents_system: UserLegalSupportAgents, 
                                           user_id: str, document_id: str) -> Dict[str, Any]:
    """Delete a specific document from the user's knowledge base."""
    return agents_system.delete_user_document(user_id, document_id)


def delete_user_session_from_knowledge_base(agents_system: UserLegalSupportAgents,
                                          user_id: str, session_id: str) -> Dict[str, Any]:
    """Delete all documents for a user's specific session."""
    return agents_system.delete_user_session_documents(user_id, session_id)


def get_user_documents_list(agents_system: UserLegalSupportAgents, user_id: str,
                          session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all documents from the user's knowledge base."""
    return agents_system.get_user_documents(user_id, session_id)


def get_all_system_documents(agents_system: UserLegalSupportAgents) -> List[Dict[str, Any]]:
    """Get all documents from all users and sessions."""
    return agents_system.get_all_documents()


def search_all_user_documents(agents_system: UserLegalSupportAgents, query: str, 
                             limit: int = 3, user_id: Optional[str] = None,
                             session_id: Optional[str] = None,
                             document_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search documents across all users."""
    return agents_system.search_all_documents(
        query=query, limit=limit, user_id=user_id,
        session_id=session_id, document_type=document_type
    )


def perform_system_health_check(agents_system: UserLegalSupportAgents) -> Dict[str, Any]:
    """Perform a comprehensive health check of the system."""
    return agents_system.health_check()


def cleanup_system_orphaned_chunks(agents_system: UserLegalSupportAgents) -> Dict[str, Any]:
    """Clean up orphaned or corrupted chunks in the system."""
    return agents_system.cleanup_orphaned_chunks()


def get_documents_by_document_type(agents_system: UserLegalSupportAgents, document_type: str) -> List[Dict[str, Any]]:
    """Retrieve all documents of a specific type across all users."""
    return agents_system.get_documents_by_type(document_type)


def similarity_search_user_documents_with_score(agents_system: UserLegalSupportAgents, user_id: str, 
                                               query: str, k: int = 3, 
                                               session_id: Optional[str] = None,
                                               document_type: Optional[str] = None) -> List[tuple]:
    """Search for similar documents and return with scores for a specific user."""
    return agents_system.similarity_search_with_score(
        user_id=user_id, query=query, k=k, session_id=session_id, document_type=document_type
    )


def search_user_documents_with_structure(agents_system: UserLegalSupportAgents, user_id: str, 
                                        query: str, limit: int = 3, 
                                        session_id: Optional[str] = None,
                                        document_type: Optional[str] = None) -> DocumentSearchResult:
    """Search documents and return structured results for a specific user."""
    return agents_system.search_user_documents_structured(
        user_id=user_id, query=query, limit=limit, 
        session_id=session_id, document_type=document_type
    )