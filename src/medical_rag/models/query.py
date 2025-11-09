"""
Модель запроса для RAG системы
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class Query:
    """Модель запроса пользователя"""
    text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    context: Optional[Dict[str, Any]] = None
    max_results: int = 3
    similarity_threshold: float = 0.5
    
    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.max_results <= 0:
            raise ValueError("Max results must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует запрос в словарь"""
        return {
            "text": self.text,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "context": self.context,
            "max_results": self.max_results,
            "similarity_threshold": self.similarity_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Query":
        """Создает запрос из словаря"""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])
        
        return cls(
            text=data["text"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            timestamp=timestamp,
            context=data.get("context"),
            max_results=data.get("max_results", 3),
            similarity_threshold=data.get("similarity_threshold", 0.5)
        )
    
    def is_medical_query(self) -> bool:
        """Проверяет, является ли запрос медицинским"""
        medical_keywords = [
            "диагноз", "симптом", "лечение", "болезнь", "заболевание",
            "варикоз", "тромбоз", "флебит", "вен", "артерий",
            "медицин", "клиническ", "патологи", "терапи"
        ]
        query_lower = self.text.lower()
        return any(keyword in query_lower for keyword in medical_keywords)
    
    def get_query_type(self) -> str:
        """Определяет тип запроса"""
        if "что такое" in self.text.lower():
            return "definition"
        elif "как" in self.text.lower() or "каким образом" in self.text.lower():
            return "procedure"
        elif "какие" in self.text.lower() or "какой" in self.text.lower():
            return "list"
        elif "когда" in self.text.lower():
            return "temporal"
        elif "где" in self.text.lower():
            return "location"
        else:
            return "general"
