"""
Configuration management for the AI-powered financial document processing system.
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/financial_docs"
    
    # OpenAI configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4-1106-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Pinecone configuration
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str = "gcp-starter"
    PINECONE_INDEX_NAME: str = "financial-documents"
    PINECONE_DIMENSION: int = 3072  # For text-embedding-3-large
    
    # Application configuration
    APP_NAME: str = "Financial Document AI System"
    DEBUG: bool = False
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_TYPES: list = [".pdf"]
    
    # Upload directory
    UPLOAD_DIRECTORY: str = "/tmp/uploads"
    
    # Chunk settings for document processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings