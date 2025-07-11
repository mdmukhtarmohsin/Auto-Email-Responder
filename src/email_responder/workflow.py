"""LangGraph workflow orchestration for email processing pipeline."""

import logging
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import asdict

from langgraph.graph import Graph, END
from langgraph.checkpoint.memory import MemorySaver

from .gmail_fetcher import gmail_fetcher, EmailData
from .retriever_chain import policy_retriever
from .llm_response_chain import llm_response_chain
from .email_sender import email_sender
from .config import config
from .cache_manager import cache_manager

logger = logging.getLogger(__name__)


class EmailProcessingState(TypedDict):
    """State structure for the email processing workflow."""
    
    # Input data
    emails: List[EmailData]
    current_email: Optional[EmailData]
    
    # Processing state
    processed_count: int
    successful_responses: int
    failed_responses: int
    
    # Current email processing
    intent: Optional[str]
    relevant_docs: List[Any]  # Documents
    generated_response: Optional[str]
    send_result: Optional[Dict[str, Any]]
    
    # Errors and logging
    last_error: Optional[str]
    processing_log: List[str]


class EmailWorkflow:
    """LangGraph workflow for automated email processing."""
    
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """Build the LangGraph workflow."""
        
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("fetch_emails", self._fetch_emails)
        workflow.add_node("process_email", self._process_single_email)
        workflow.add_node("retrieve_policies", self._retrieve_policies)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("send_response", self._send_response)
        workflow.add_node("next_email", self._next_email)
        workflow.add_node("finalize", self._finalize_processing)
        
        # Define workflow edges
        workflow.set_entry_point("fetch_emails")
        
        # From fetch_emails
        workflow.add_conditional_edges(
            "fetch_emails",
            self._should_process_emails,
            {
                "process": "process_email",
                "complete": "finalize"
            }
        )
        
        # Email processing chain
        workflow.add_edge("process_email", "retrieve_policies")
        workflow.add_edge("retrieve_policies", "generate_response")
        workflow.add_edge("generate_response", "send_response")
        
        # After sending response
        workflow.add_conditional_edges(
            "send_response",
            self._should_continue_processing,
            {
                "continue": "next_email",
                "complete": "finalize"
            }
        )
        
        # Continue with next email
        workflow.add_edge("next_email", "process_email")
        
        # End workflow
        workflow.add_edge("finalize", END)
        
        # Compile workflow with memory
        self.graph = workflow.compile(checkpointer=self.memory)
        
        logger.info("Email processing workflow compiled successfully")
    
    def _fetch_emails(self, state: EmailProcessingState) -> EmailProcessingState:
        """Fetch unread emails from Gmail."""
        
        logger.info("Starting email fetch process")
        
        try:
            # Fetch unread emails
            emails = gmail_fetcher.fetch_unread_emails(limit=config.max_emails_per_batch)
            
            state["emails"] = emails
            state["processed_count"] = 0
            state["successful_responses"] = 0
            state["failed_responses"] = 0
            state["processing_log"] = [f"Fetched {len(emails)} unread emails"]
            
            logger.info(f"Fetched {len(emails)} unread emails for processing")
            
        except Exception as e:
            error_msg = f"Failed to fetch emails: {e}"
            logger.error(error_msg)
            state["last_error"] = error_msg
            state["emails"] = []
            state["processing_log"] = [error_msg]
        
        return state
    
    def _should_process_emails(self, state: EmailProcessingState) -> str:
        """Determine if there are emails to process."""
        emails = state.get("emails", [])
        if emails and len(emails) > 0:
            return "process"
        return "complete"
    
    def _process_single_email(self, state: EmailProcessingState) -> EmailProcessingState:
        """Process a single email from the queue."""
        
        emails = state.get("emails", [])
        processed_count = state.get("processed_count", 0)
        
        if processed_count < len(emails):
            current_email = emails[processed_count]
            state["current_email"] = current_email
            
            log_msg = f"Processing email {processed_count + 1}/{len(emails)}: {current_email.subject}"
            logger.info(log_msg)
            
            processing_log = state.get("processing_log", [])
            processing_log.append(log_msg)
            state["processing_log"] = processing_log
            
            # Mark email as being processed
            try:
                gmail_fetcher.mark_as_read(current_email.message_id)
            except Exception as e:
                logger.warning(f"Failed to mark email as read: {e}")
        
        return state
    
    def _retrieve_policies(self, state: EmailProcessingState) -> EmailProcessingState:
        """Retrieve relevant policy documents for the current email."""
        
        current_email = state.get("current_email")
        if not current_email:
            state["last_error"] = "No current email to process"
            return state
        
        try:
            # Retrieve relevant documents and intent
            relevant_docs, intent = policy_retriever.retrieve_relevant_policies(
                email_subject=current_email.subject,
                email_body=current_email.body
            )
            
            state["relevant_docs"] = relevant_docs
            state["intent"] = intent
            
            log_msg = f"Retrieved {len(relevant_docs)} relevant documents for intent: {intent}"
            logger.info(log_msg)
            
            processing_log = state.get("processing_log", [])
            processing_log.append(log_msg)
            state["processing_log"] = processing_log
            
        except Exception as e:
            error_msg = f"Failed to retrieve policies: {e}"
            logger.error(error_msg)
            state["last_error"] = error_msg
            state["relevant_docs"] = []
            state["intent"] = "general"
        
        return state
    
    def _generate_response(self, state: EmailProcessingState) -> EmailProcessingState:
        """Generate email response using LLM."""
        
        current_email = state.get("current_email")
        relevant_docs = state.get("relevant_docs", [])
        
        if not current_email:
            state["last_error"] = "No current email to generate response for"
            return state
        
        try:
            # Generate response using LLM chain
            response = llm_response_chain.generate_response(
                email_subject=current_email.subject,
                email_body=current_email.body,
                sender_name=current_email.sender_name,
                context_documents=relevant_docs
            )
            
            state["generated_response"] = response
            
            if response:
                log_msg = f"Generated response ({len(response)} chars)"
                logger.info(log_msg)
            else:
                log_msg = "Failed to generate response"
                logger.warning(log_msg)
                state["last_error"] = "LLM failed to generate response"
            
            processing_log = state.get("processing_log", [])
            processing_log.append(log_msg)
            state["processing_log"] = processing_log
            
        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            logger.error(error_msg)
            state["last_error"] = error_msg
            state["generated_response"] = None
        
        return state
    
    def _send_response(self, state: EmailProcessingState) -> EmailProcessingState:
        """Send the generated email response."""
        
        current_email = state.get("current_email")
        generated_response = state.get("generated_response")
        
        if not current_email or not generated_response:
            error_msg = "Missing email or response for sending"
            state["last_error"] = error_msg
            state["failed_responses"] = state.get("failed_responses", 0) + 1
            return state
        
        try:
            # Send email response
            success = email_sender.send_response_email(
                to_email=current_email.sender_email,
                subject=current_email.subject,
                body=generated_response,
                original_message_id=current_email.message_id,
                thread_id=current_email.thread_id
            )
            
            if success:
                state["successful_responses"] = state.get("successful_responses", 0) + 1
                log_msg = f"Successfully sent response to {current_email.sender_email}"
                logger.info(log_msg)
            else:
                state["failed_responses"] = state.get("failed_responses", 0) + 1
                log_msg = f"Failed to send response to {current_email.sender_email}"
                logger.error(log_msg)
                state["last_error"] = "Email sending failed"
            
            processing_log = state.get("processing_log", [])
            processing_log.append(log_msg)
            state["processing_log"] = processing_log
            
            state["send_result"] = {"success": success}
            
        except Exception as e:
            error_msg = f"Error sending response: {e}"
            logger.error(error_msg)
            state["last_error"] = error_msg
            state["failed_responses"] = state.get("failed_responses", 0) + 1
        
        return state
    
    def _next_email(self, state: EmailProcessingState) -> EmailProcessingState:
        """Move to the next email in the queue."""
        
        processed_count = state.get("processed_count", 0)
        state["processed_count"] = processed_count + 1
        
        # Clear current email state
        state["current_email"] = None
        state["intent"] = None
        state["relevant_docs"] = []
        state["generated_response"] = None
        state["send_result"] = None
        state["last_error"] = None
        
        return state
    
    def _should_continue_processing(self, state: EmailProcessingState) -> str:
        """Determine if there are more emails to process."""
        
        emails = state.get("emails", [])
        processed_count = state.get("processed_count", 0)
        
        if processed_count < len(emails):
            return "continue"
        return "complete"
    
    def _finalize_processing(self, state: EmailProcessingState) -> EmailProcessingState:
        """Finalize the processing session."""
        
        successful = state.get("successful_responses", 0)
        failed = state.get("failed_responses", 0)
        total_emails = len(state.get("emails", []))
        
        summary = f"Processing complete: {successful} successful, {failed} failed out of {total_emails} emails"
        logger.info(summary)
        
        processing_log = state.get("processing_log", [])
        processing_log.append(summary)
        state["processing_log"] = processing_log
        
        return state
    
    def process_emails(self, thread_id: str = "default") -> Dict[str, Any]:
        """Process emails using the workflow.
        
        Args:
            thread_id: Unique thread ID for this processing session
            
        Returns:
            Dict containing processing results and statistics
        """
        
        if not self.graph:
            logger.error("Workflow graph not initialized")
            return {"error": "Workflow not initialized"}
        
        try:
            # Initialize state
            initial_state: EmailProcessingState = {
                "emails": [],
                "current_email": None,
                "processed_count": 0,
                "successful_responses": 0,
                "failed_responses": 0,
                "intent": None,
                "relevant_docs": [],
                "generated_response": None,
                "send_result": None,
                "last_error": None,
                "processing_log": []
            }
            
            # Run workflow
            config_dict = {"configurable": {"thread_id": thread_id}}
            final_state = self.graph.invoke(initial_state, config=config_dict)
            
            # Return results
            return {
                "success": True,
                "total_emails": len(final_state.get("emails", [])),
                "successful_responses": final_state.get("successful_responses", 0),
                "failed_responses": final_state.get("failed_responses", 0),
                "processing_log": final_state.get("processing_log", []),
                "last_error": final_state.get("last_error")
            }
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the status of workflow components."""
        
        return {
            "workflow_ready": self.graph is not None,
            "gmail_fetcher_ready": gmail_fetcher.test_connection(),
            "policy_retriever_ready": policy_retriever.get_index_stats()["vectorstore_exists"],
            "llm_response_ready": llm_response_chain.test_connection(),
            "email_sender_ready": email_sender.test_connection(),
            "cache_ready": cache_manager.test_connection()
        }


# Global workflow instance
email_workflow = EmailWorkflow() 