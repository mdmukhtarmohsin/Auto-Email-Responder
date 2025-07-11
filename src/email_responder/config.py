"""Configuration management for the email responder system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config(BaseSettings):
    """Application configuration."""
    
    # Gemini API Configuration
    google_api_key: str = Field(default="", description="Gemini API key")
    gemini_model: str = Field(default="gemini-2.0-flash-exp", description="Gemini model name")
    
    # Gmail API Configuration
    gmail_credentials_path: str = Field(default="credentials.json", description="Gmail credentials file path")
    gmail_token_path: str = Field(default="token.json", description="Gmail token file path")
    gmail_email_address: str = Field(default="", description="Gmail email address")
    
    # Vector Database Configuration
    vector_db_type: str = Field(default="chroma", description="Vector database type")
    vector_db_path: str = Field(default="./data/chroma_db", description="Vector database path")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name")
    
    # Document Processing Configuration
    chunk_size: int = Field(default=1000, description="Document chunk size for splitting")
    chunk_overlap: int = Field(default=200, description="Document chunk overlap")
    top_k_docs: int = Field(default=5, description="Number of top documents to retrieve")
    policies_dir: str = Field(default="./config/policies", description="Directory containing policy documents")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    use_redis_cache: bool = Field(default=False, description="Use Redis for caching")
    
    # Application Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    batch_size: int = Field(default=10, description="Email batch size")
    processing_interval_minutes: int = Field(default=5, description="Processing interval in minutes")
    max_response_length: int = Field(default=500, description="Maximum response length")
    
    # Safety and Content Configuration
    enable_safety_checks: bool = Field(default=True, description="Enable safety checks")
    response_tone: str = Field(default="polite", description="Response tone")
    
    # Performance Configuration
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours")
    max_emails_per_batch: int = Field(default=50, description="Maximum emails per batch")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_prefix": "",
    }


# Global config instance - wrap in try-catch for missing required fields
try:
    config = Config()
except Exception as e:
    print(f"Warning: Configuration error: {e}")
    print("Please check your .env file or environment variables.")
    config = Config(
        google_api_key="",
        gmail_email_address=""
    ) 