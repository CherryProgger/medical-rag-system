"""
Medical RAG System
Профессиональная RAG система для медицинской документации
"""

__version__ = "1.0.0"
__author__ = "Medical RAG Team"
__email__ = "medical-rag@example.com"

from .core.rag_system import MedicalRAGSystem
from .core.embedding_service import EmbeddingService
from .core.retrieval_service import RetrievalService
from .core.generation_service import GenerationService
from .data.data_processor import DataProcessor
from .models.document import Document
from .models.query import Query
from .models.response import Response

__all__ = [
    "MedicalRAGSystem",
    "EmbeddingService", 
    "RetrievalService",
    "GenerationService",
    "DataProcessor",
    "Document",
    "Query", 
    "Response"
]
