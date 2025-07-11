"""Policy Retriever Chain using LangChain and vector search."""

import logging
import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from .config import config
from .cache_manager import cache_manager

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifies email intent to route to appropriate policies."""
    
    INTENTS = {
        "billing": ["billing", "payment", "invoice", "refund", "subscription", "charge", "cost", "price"],
        "technical_support": ["error", "bug", "problem", "issue", "broken", "not working", "help", "technical"],
        "feature_request": ["feature", "request", "suggestion", "improvement", "new", "add", "enhancement"],
        "general": ["question", "info", "information", "contact", "hours", "address", "about"]
    }
    
    def __init__(self):
        self.llm = None
        if config.google_api_key:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash-exp",
                    google_api_key=config.google_api_key,
                    temperature=0.1
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLM for intent classification: {e}")
    
    def classify_intent(self, email_subject: str, email_body: str) -> str:
        """Classify email intent using keyword matching and optional LLM."""
        
        # Combine subject and body for analysis
        text = f"{email_subject} {email_body}".lower()
        
        # Keyword-based classification
        intent_scores = {}
        for intent, keywords in self.INTENTS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent
        if intent_scores:
            classified_intent = max(intent_scores, key=intent_scores.get)
            logger.info(f"Classified intent as: {classified_intent}")
            return classified_intent
        
        # Fallback to LLM if no keywords match
        if self.llm:
            try:
                prompt = f"""Classify the following email into one of these categories:
- billing: Payment, subscription, refund, invoice issues
- technical_support: Technical problems, bugs, errors
- feature_request: New feature requests, suggestions
- general: General questions, information requests

Email Subject: {email_subject}
Email Body: {email_body}

Respond with only the category name:"""
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                llm_intent = response.content.strip().lower()
                
                if llm_intent in self.INTENTS:
                    logger.info(f"LLM classified intent as: {llm_intent}")
                    return llm_intent
                    
            except Exception as e:
                logger.warning(f"LLM intent classification failed: {e}")
        
        # Default fallback
        logger.info("Using default intent: general")
        return "general"


class PolicyRetriever:
    """Retrieves relevant policy documents using vector search."""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.text_splitter = None
        self.intent_classifier = IntentClassifier()
        self._setup_embeddings()
        self._setup_text_splitter()
        self._initialize_vectorstore()
    
    def _setup_embeddings(self) -> None:
        """Setup HuggingFace embeddings."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Initialized HuggingFace embeddings")
        except Exception as e:
            logger.error(f"Failed to setup embeddings: {e}")
    
    def _setup_text_splitter(self) -> None:
        """Setup text splitter for document chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _initialize_vectorstore(self) -> None:
        """Initialize or load existing vectorstore."""
        persist_directory = config.vector_db_path
        
        try:
            if os.path.exists(persist_directory) and os.listdir(persist_directory):
                # Load existing vectorstore
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vectorstore from {persist_directory}")
            else:
                # Create new vectorstore
                self._build_vectorstore()
            
            if self.vectorstore:
                self._setup_retriever()
                
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}")
    
    def _build_vectorstore(self) -> None:
        """Build vectorstore from policy documents."""
        if not self.embeddings:
            logger.error("Cannot build vectorstore: embeddings not initialized")
            return
        
        documents = self._load_policy_documents()
        if not documents:
            logger.warning("No policy documents found")
            return
        
        # Split documents into chunks
        chunked_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
        
        try:
            # Create vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=chunked_docs,
                embedding=self.embeddings,
                persist_directory=config.vector_db_path
            )
            
            # Persist the vectorstore
            self.vectorstore.persist()
            logger.info(f"Built and persisted vectorstore to {config.vector_db_path}")
            
        except Exception as e:
            logger.error(f"Failed to build vectorstore: {e}")
    
    def _load_policy_documents(self) -> List[Document]:
        """Load policy documents from markdown files."""
        documents = []
        policies_dir = Path(config.policies_dir)
        
        if not policies_dir.exists():
            logger.error(f"Policies directory not found: {policies_dir}")
            return documents
        
        for md_file in policies_dir.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(md_file),
                        "filename": md_file.name,
                        "type": "policy"
                    }
                )
                documents.append(doc)
                logger.debug(f"Loaded policy document: {md_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to load {md_file}: {e}")
        
        logger.info(f"Loaded {len(documents)} policy documents")
        return documents
    
    def _setup_retriever(self) -> None:
        """Setup multi-query retriever."""
        if not self.vectorstore:
            logger.error("Cannot setup retriever: vectorstore not initialized")
            return
        
        try:
            # Base retriever
            base_retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": config.top_k_docs}
            )
            
            # Multi-query retriever for enhanced results
            if self.intent_classifier.llm:
                self.retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=self.intent_classifier.llm
                )
                logger.info("Setup multi-query retriever with LLM")
            else:
                self.retriever = base_retriever
                logger.info("Setup basic similarity retriever")
                
        except Exception as e:
            logger.error(f"Failed to setup retriever: {e}")
            # Fallback to basic retriever
            self.retriever = self.vectorstore.as_retriever()
    
    def retrieve_relevant_policies(
        self,
        email_subject: str,
        email_body: str,
        intent: Optional[str] = None
    ) -> Tuple[List[Document], str]:
        """Retrieve relevant policy documents for an email."""
        
        if not self.retriever:
            logger.error("Retriever not initialized")
            return [], "general"
        
        # Classify intent if not provided
        if not intent:
            intent = self.intent_classifier.classify_intent(email_subject, email_body)
        
        # Create search query
        query = f"Subject: {email_subject}\nContent: {email_body}"
        
        # Check cache first
        cache_key = f"retrieval_{intent}_{hash(query) % 10000}"
        cached_docs = cache_manager.get_embeddings(cache_key)
        if cached_docs:
            logger.debug("Retrieved documents from cache")
            return cached_docs, intent
        
        try:
            # Retrieve documents
            documents = self.retriever.invoke(query)
            
            # Filter and rank documents by intent relevance
            ranked_docs = self._rank_documents_by_intent(documents, intent)
            
            # Cache the results
            cache_manager.set_embeddings(cache_key, ranked_docs)
            
            logger.info(f"Retrieved {len(ranked_docs)} relevant documents for intent: {intent}")
            return ranked_docs, intent
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return [], intent
    
    def _rank_documents_by_intent(self, documents: List[Document], intent: str) -> List[Document]:
        """Rank documents by relevance to classified intent."""
        if not documents:
            return documents
        
        # Intent-based keywords for ranking
        intent_keywords = self.intent_classifier.INTENTS.get(intent, [])
        
        # Score documents based on intent keyword matches
        scored_docs = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for keyword in intent_keywords if keyword in content_lower)
            
            # Boost score if filename matches intent
            filename = doc.metadata.get("filename", "").lower()
            if any(keyword in filename for keyword in intent_keywords):
                score += 2
            
            scored_docs.append((score, doc))
        
        # Sort by score (descending) and return documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top documents, ensuring we don't exceed the limit
        max_docs = min(config.top_k_docs, len(scored_docs))
        return [doc for _, doc in scored_docs[:max_docs]]
    
    def refresh_index(self) -> bool:
        """Refresh the vector index with latest policy documents."""
        try:
            # Remove existing vectorstore
            if os.path.exists(config.vector_db_path):
                import shutil
                shutil.rmtree(config.vector_db_path)
                logger.info("Removed existing vectorstore")
            
            # Rebuild vectorstore
            self._build_vectorstore()
            
            if self.vectorstore:
                self._setup_retriever()
                cache_manager.clear_embeddings_cache()
                logger.info("Successfully refreshed policy index")
                return True
                
        except Exception as e:
            logger.error(f"Failed to refresh index: {e}")
            
        return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        stats = {
            "vectorstore_exists": self.vectorstore is not None,
            "retriever_ready": self.retriever is not None,
            "embeddings_ready": self.embeddings is not None,
            "index_path": config.vector_db_path,
            "policies_dir": config.policies_dir
        }
        
        if self.vectorstore:
            try:
                # Try to get collection info if available
                collection = self.vectorstore._collection
                stats["document_count"] = collection.count()
            except:
                stats["document_count"] = "unknown"
        
        return stats


# Global policy retriever instance
policy_retriever = PolicyRetriever() 