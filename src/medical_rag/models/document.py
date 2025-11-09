"""
Модель документа для RAG системы
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DocumentMetadata:
    """Метаданные документа"""

    category: str = ""
    difficulty: str = ""
    topic: str = ""
    source_file: str = ""
    section: Optional[str] = None
    confidence: str = "high"
    created_at: datetime = field(default_factory=datetime.now)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метаданные в словарь"""
        return {
            "category": self.category,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "source_file": self.source_file,
            "section": self.section,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Создает метаданные из словаря"""
        return cls(
            category=data.get("category", ""),
            difficulty=data.get("difficulty", ""),
            topic=data.get("topic", ""),
            source_file=data.get("source_file", ""),
            section=data.get("section"),
            confidence=data.get("confidence", "high"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            extra=data.get("extra", {}),
        )


@dataclass
class Document:
    """Модель документа в RAG системе"""

    id: str
    content: str
    question: str
    answer: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")
        if not self.question:
            raise ValueError("Document question cannot be empty")
        if not self.answer:
            raise ValueError("Document answer cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует документ в словарь"""
        return {
            "id": self.id,
            "content": self.content,
            "question": self.question,
            "answer": self.answer,
            "metadata": self.metadata.to_dict(),
            "embedding": self.embedding,
            "similarity_score": self.similarity_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Создает документ из словаря"""
        return cls(
            id=data["id"],
            content=data["content"],
            question=data["question"],
            answer=data["answer"],
            metadata=DocumentMetadata.from_dict(data.get("metadata", {})),
            embedding=data.get("embedding"),
            similarity_score=data.get("similarity_score"),
        )

    def get_source_info(self) -> str:
        """Возвращает информацию об источнике"""
        source_info = f"Файл: {self.metadata.source_file}"
        if self.metadata.section:
            source_info += f", Раздел: {self.metadata.section}"
        if "chunk_index" in self.metadata.extra:
            source_info += f", Чанк: {self.metadata.extra['chunk_index']}"
        return source_info

    def is_relevant(self, threshold: float = 0.5) -> bool:
        """Проверяет релевантность документа"""
        return self.similarity_score is not None and self.similarity_score >= threshold

    def get_confidence_level(self) -> str:
        """Возвращает уровень уверенности"""
        if self.similarity_score is None:
            return "unknown"
        if self.similarity_score >= 0.8:
            return "high"
        if self.similarity_score >= 0.6:
            return "medium"
        return "low"

    def get_cache_key(self) -> str:
        """Возвращает ключ кэша эмбеддинга."""
        if self.metadata.extra and "hash" in self.metadata.extra:
            return str(self.metadata.extra["hash"])
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
