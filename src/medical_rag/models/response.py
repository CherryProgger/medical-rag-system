"""
Модель ответа для RAG системы
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .document import Document


@dataclass
class ResponseMetadata:
    """Метаданные ответа"""
    processing_time: float
    num_documents_searched: int
    num_documents_found: int
    best_similarity_score: float
    confidence_level: str
    query_type: str
    is_medical_query: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метаданные в словарь"""
        return {
            "processing_time": self.processing_time,
            "num_documents_searched": self.num_documents_searched,
            "num_documents_found": self.num_documents_found,
            "best_similarity_score": self.best_similarity_score,
            "confidence_level": self.confidence_level,
            "query_type": self.query_type,
            "is_medical_query": self.is_medical_query,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Response:
    """Модель ответа RAG системы"""
    query: str
    answer: str
    relevant_documents: List[Document]
    metadata: ResponseMetadata
    sources: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.query:
            raise ValueError("Response query cannot be empty")
        if not self.answer:
            raise ValueError("Response answer cannot be empty")
        if not isinstance(self.relevant_documents, list):
            raise ValueError("Relevant documents must be a list")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует ответ в словарь"""
        return {
            "query": self.query,
            "answer": self.answer,
            "relevant_documents": [doc.to_dict() for doc in self.relevant_documents],
            "metadata": self.metadata.to_dict(),
            "sources": self.sources,
            "warnings": self.warnings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        """Создает ответ из словаря"""
        relevant_docs = [Document.from_dict(doc) for doc in data.get("relevant_documents", [])]
        
        metadata_data = data.get("metadata", {})
        metadata = ResponseMetadata(
            processing_time=metadata_data.get("processing_time", 0.0),
            num_documents_searched=metadata_data.get("num_documents_searched", 0),
            num_documents_found=metadata_data.get("num_documents_found", 0),
            best_similarity_score=metadata_data.get("best_similarity_score", 0.0),
            confidence_level=metadata_data.get("confidence_level", "unknown"),
            query_type=metadata_data.get("query_type", "general"),
            is_medical_query=metadata_data.get("is_medical_query", False),
            timestamp=datetime.fromisoformat(metadata_data.get("timestamp", datetime.now().isoformat()))
        )
        
        return cls(
            query=data["query"],
            answer=data["answer"],
            relevant_documents=relevant_docs,
            metadata=metadata,
            sources=data.get("sources", []),
            warnings=data.get("warnings", [])
        )
    
    def get_confidence_level(self) -> str:
        """Возвращает уровень уверенности ответа"""
        return self.metadata.confidence_level
    
    def is_high_confidence(self) -> bool:
        """Проверяет, является ли ответ высокоуверенным"""
        return self.metadata.confidence_level == "high"
    
    def get_source_summary(self) -> str:
        """Возвращает краткую сводку источников"""
        if not self.relevant_documents:
            return "Источники не найдены"
        
        sources = []
        for doc in self.relevant_documents[:3]:  # Топ-3 источника
            source_info = doc.get_source_info()
            sources.append(f"• {source_info} (релевантность: {doc.similarity_score:.3f})")
        
        return "\n".join(sources)
    
    def get_performance_summary(self) -> str:
        """Возвращает сводку производительности"""
        return (
            f"Время обработки: {self.metadata.processing_time:.2f}с | "
            f"Найдено документов: {self.metadata.num_documents_found} | "
            f"Лучший score: {self.metadata.best_similarity_score:.3f}"
        )
    
    def add_warning(self, warning: str):
        """Добавляет предупреждение"""
        self.warnings.append(warning)
    
    def has_warnings(self) -> bool:
        """Проверяет наличие предупреждений"""
        return len(self.warnings) > 0
