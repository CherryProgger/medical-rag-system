"""
Сервис для создания эмбеддингов
"""

import numpy as np
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import logging
from ..models.config import Config
from ..models.document import Document

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Сервис для создания и работы с эмбеддингами"""
    
    def __init__(self, config: Config):
        """
        Инициализация сервиса эмбеддингов
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализирует модель для создания эмбеддингов"""
        try:
            logger.info(f"Загружаем модель эмбеддингов: {self.config.model.embedding_model}")
            self.model = SentenceTransformer(self.config.model.embedding_model)
            logger.info("Модель эмбеддингов успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов: {e}")
            raise
    
    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Создает эмбеддинги для списка текстов
        
        Args:
            texts: Список текстов для создания эмбеддингов
            batch_size: Размер батча для обработки
            
        Returns:
            Массив эмбеддингов
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Создаем эмбеддинги для {len(texts)} текстов")
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Создано {len(embeddings)} эмбеддингов размерности {embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            raise
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Создает эмбеддинг для одного текста
        
        Args:
            text: Текст для создания эмбеддинга
            
        Returns:
            Эмбеддинг текста
        """
        return self.create_embeddings([text])[0]
    
    def create_document_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Создает эмбеддинги для документов
        
        Args:
            documents: Список документов
            
        Returns:
            Документы с добавленными эмбеддингами
        """
        if not documents:
            return documents
        
        # Извлекаем тексты для создания эмбеддингов
        texts = [doc.content for doc in documents]
        
        # Создаем эмбеддинги
        embeddings = self.create_embeddings(texts)
        
        # Добавляем эмбеддинги к документам
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding.tolist()
        
        logger.info(f"Созданы эмбеддинги для {len(documents)} документов")
        return documents
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычисляет косинусное сходство между двумя эмбеддингами
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
            
        Returns:
            Косинусное сходство
        """
        try:
            # Нормализуем эмбеддинги
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Вычисляем косинусное сходство
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Ошибка вычисления сходства: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, document_embeddings: List[np.ndarray]) -> tuple:
        """
        Находит наиболее похожий документ
        
        Args:
            query_embedding: Эмбеддинг запроса
            document_embeddings: Список эмбеддингов документов
            
        Returns:
            Кортеж (индекс, сходство) наиболее похожего документа
        """
        if not document_embeddings:
            return -1, 0.0
        
        similarities = []
        for doc_embedding in document_embeddings:
            similarity = self.compute_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        
        return max_index, max_similarity
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Нормализует эмбеддинги для косинусного сходства
        
        Args:
            embeddings: Массив эмбеддингов
            
        Returns:
            Нормализованные эмбеддинги
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Избегаем деления на ноль
        return embeddings / norms
    
    def get_embedding_dimension(self) -> int:
        """Возвращает размерность эмбеддингов"""
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель"""
        return self.model is not None
