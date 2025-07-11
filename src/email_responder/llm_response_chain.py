"""LLM response chain using Gemini for email response generation."""

import logging
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import BaseOutputParser

from .config import config
from .cache_manager import cache_manager

logger = logging.getLogger(__name__)


class EmailResponseParser(BaseOutputParser):
    """Custom parser for email responses."""
    
    def parse(self, text: str) -> str:
        """Parse the LLM response to extract clean email content."""
        # Remove any system messages or instructions
        lines = text.strip().split('\n')
        clean_lines = []
        
        for line in lines:
            # Skip lines that look like instructions
            if line.startswith('[') and line.endswith(']'):
                continue
            if 'assistant:' in line.lower() or 'system:' in line.lower():
                continue
            clean_lines.append(line)
        
        response = '\n'.join(clean_lines).strip()
        
        # Limit response length
        if len(response) > config.max_response_length:
            response = response[:config.max_response_length] + "..."
        
        return response


class LLMResponseChain:
    """LLM chain for generating email responses using Gemini."""
    
    def __init__(self):
        self.llm = None
        self.prompt_template = None
        self.chain = None
        self.parser = EmailResponseParser()
        self._setup_llm()
        self._setup_prompt()
        self._setup_chain()
    
    def _setup_llm(self) -> None:
        """Setup Gemini LLM."""
        try:
            if not config.google_api_key:
                logger.error("Google API key not configured")
                return
            
            self.llm = ChatGoogleGenerativeAI(
                model=config.gemini_model,
                google_api_key=config.google_api_key,
                temperature=0.3,
                max_tokens=config.max_response_length,
                top_p=0.8,
                top_k=40
            )
            
            logger.info(f"Initialized Gemini model: {config.gemini_model}")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
    
    def _setup_prompt(self) -> None:
        """Setup prompt template for email responses."""
        
        system_prompt = """You are a helpful customer service AI assistant. Your job is to generate professional, {tone}, and helpful email responses based on the customer's inquiry and relevant company policies.

Guidelines:
1. Be {tone} and professional in tone
2. Address the customer's specific question or concern
3. Use the provided policy information to give accurate answers
4. Keep responses concise but complete
5. If you cannot find relevant information, politely direct them to human support
6. Do not make up information not contained in the policies
7. Always end with a professional closing

Policy Information:
{context}

Customer Email:
Subject: {email_subject}
From: {sender_name}
Content: {email_body}

Please generate a professional email response:"""

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Generate an appropriate response to this customer email.")
        ])
    
    def _setup_chain(self) -> None:
        """Setup the LangChain response chain."""
        if not self.llm or not self.prompt_template:
            logger.error("Cannot setup chain: LLM or prompt not initialized")
            return
        
        try:
            self.chain = (
                RunnablePassthrough.assign(
                    context=lambda x: self._format_context(x.get("context", [])),
                    tone=lambda x: config.response_tone
                )
                | self.prompt_template
                | self.llm
                | self.parser
            )
            
            logger.info("Response chain setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup response chain: {e}")
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context."""
        if not documents:
            return "No specific policy information available for this query."
        
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 documents
            # Extract filename from metadata if available
            source = doc.metadata.get('source', f'Document {i}')
            if source:
                filename = source.split('/')[-1].replace('.md', '')
                context_parts.append(f"## {filename.title()}\n{doc.page_content}")
            else:
                context_parts.append(f"## Policy Information {i}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
    
    def _create_cache_key(self, email_subject: str, email_body: str, context_docs: List[Document]) -> str:
        """Create cache key for response."""
        # Create a hash based on email content and context
        content = f"{email_subject}|{email_body}"
        if context_docs:
            context_content = "|".join([doc.page_content[:100] for doc in context_docs[:3]])
            content += f"|{context_content}"
        return content
    
    def generate_response(
        self,
        email_subject: str,
        email_body: str,
        sender_name: str,
        context_documents: List[Document]
    ) -> Optional[str]:
        """Generate email response using the LLM chain."""
        
        if not self.chain:
            logger.error("Response chain not initialized")
            return None
        
        # Check cache first
        cache_key = self._create_cache_key(email_subject, email_body, context_documents)
        cached_response = cache_manager.get_prompt_response(cache_key)
        if cached_response:
            logger.debug("Retrieved response from cache")
            return cached_response
        
        try:
            # Prepare input data
            input_data = {
                "email_subject": email_subject,
                "email_body": email_body,
                "sender_name": sender_name,
                "context": context_documents
            }
            
            # Generate response
            response = self.chain.invoke(input_data)
            
            # Validate response
            if not response or len(response.strip()) < 10:
                logger.warning("Generated response too short, using fallback")
                return self._generate_fallback_response(sender_name)
            
            # Cache the response
            cache_manager.set_prompt_response(cache_key, response)
            
            logger.info("Generated email response successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(sender_name)
    
    def _generate_fallback_response(self, sender_name: str) -> str:
        """Generate a fallback response when LLM fails."""
        return f"""Dear {sender_name},

Thank you for contacting us. I've received your email and want to ensure you get the best possible assistance.

I'm currently experiencing some technical difficulties in processing your specific request. To ensure you receive accurate and timely support, I'm forwarding your inquiry to our customer service team.

You can expect a response from our team within 24 hours during business hours (Monday-Friday, 9 AM-6 PM EST).

For urgent matters, please contact us directly at:
- Phone: 1-800-555-0123
- Email: support@company.com

Thank you for your patience and understanding.

Best regards,
Customer Service Team"""
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        if not self.llm:
            return False
        
        try:
            response = self.llm.invoke([HumanMessage(content="Hello, please respond with 'OK'")])
            return "OK" in response.content
        except Exception as e:
            logger.error(f"LLM connection test failed: {e}")
            return False


# Global LLM response chain instance
llm_response_chain = LLMResponseChain() 