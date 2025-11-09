"""
Модели данных для медицинской RAG системы
"""

from .document import Document
from .query import Query
from .response import Response
from .config import Config

__all__ = ["Document", "Query", "Response", "Config"]
