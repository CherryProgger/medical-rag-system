"""
Сервис для поиска релевантных документов
"""

import numpy as np
import faiss
from typing import List, Optional, Tuple
import logging
from ..models.config import Config
from ..models.document import Document
from ..models.query import Query
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievalService:
    """Сервис для поиска релевантных документов"""
    
    def __init__(self, config: Config, embedding_service: EmbeddingService):
        """
        Инициализация сервиса поиска
        
        Args:
            config: Конфигурация системы
            embedding_service: Сервис эмбеддингов
        """
        self.config = config
        self.embedding_service = embedding_service
        self.index = None
        self.documents = []
        self._initialize_index()
    
    def _initialize_index(self):
        """Инициализирует FAISS индекс"""
        try:
            dimension = self.embedding_service.get_embedding_dimension()
            if dimension == 0:
                raise ValueError("Embedding dimension is 0")
            
            # Создаем индекс для косинусного сходства
            self.index = faiss.IndexFlatIP(dimension)
            logger.info(f"Создан FAISS индекс с размерностью {dimension}")
        except Exception as e:
            logger.error(f"Ошибка создания FAISS индекса: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Добавляет документы в индекс
        
        Args:
            documents: Список документов для добавления
        """
        if not documents:
            logger.warning("Пустой список документов")
            return
        
        try:
            # Создаем эмбеддинги для документов
            documents_with_embeddings = self.embedding_service.create_document_embeddings(documents)
            
            # Извлекаем эмбеддинги
            embeddings = []
            for doc in documents_with_embeddings:
                if doc.embedding is not None:
                    embeddings.append(doc.embedding)
                else:
                    logger.warning(f"Документ {doc.id} не имеет эмбеддинга")
                    continue
            
            if not embeddings:
                logger.warning("Нет валидных эмбеддингов для добавления")
                return
            
            # Преобразуем в numpy массив
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Нормализуем для косинусного сходства
            embeddings_normalized = self.embedding_service.normalize_embeddings(embeddings_array)
            
            # Добавляем в индекс
            self.index.add(embeddings_normalized)
            
            # Сохраняем документы
            self.documents.extend(documents_with_embeddings)
            
            logger.info(f"Добавлено {len(embeddings)} документов в индекс")
            
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise
    
    def search(self, query: Query) -> List[Document]:
        """
        Ищет релевантные документы для запроса
        
        Args:
            query: Запрос пользователя
            
        Returns:
            Список релевантных документов
        """
        if not self.documents:
            logger.warning("Индекс пуст")
            return []
        
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.embedding_service.create_embedding(query.text)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Нормализуем для косинусного сходства
            query_embedding = self.embedding_service.normalize_embeddings(query_embedding)
            
            # Поиск в индексе
            scores, indices = self.index.search(query_embedding, query.max_results)
            
            # Фильтруем по порогу сходства
            relevant_documents = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents) and score >= query.similarity_threshold:
                    doc = self.documents[idx]
                    doc.similarity_score = float(score)
                    relevant_documents.append(doc)
            
            logger.info(f"Найдено {len(relevant_documents)} релевантных документов")
            return relevant_documents
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    def search_by_similarity(self, query_text: str, top_k: int = None, threshold: float = None) -> List[Document]:
        """
        Упрощенный поиск по тексту запроса
        
        Args:
            query_text: Текст запроса
            top_k: Количество результатов
            threshold: Порог сходства
            
        Returns:
            Список релевантных документов
        """
        if top_k is None:
            top_k = self.config.retrieval.top_k
        if threshold is None:
            threshold = self.config.retrieval.similarity_threshold
        
        query = Query(
            text=query_text,
            max_results=top_k,
            similarity_threshold=threshold
        )
        
        return self.search(query)
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Находит документ по ID
        
        Args:
            doc_id: ID документа
            
        Returns:
            Документ или None
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_statistics(self) -> dict:
        """
        Возвращает статистику индекса
        
        Returns:
            Словарь со статистикой
        """
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_service.get_embedding_dimension(),
            "index_type": "FAISS_FlatIP"
        }
    
    def clear_index(self):
        """Очищает индекс и документы"""
        if self.index:
            self.index.reset()
        self.documents.clear()
        logger.info("Индекс очищен")
    
    def rebuild_index(self, documents: List[Document]):
        """
        Перестраивает индекс с новыми документами
        
        Args:
            documents: Новые документы
        """
        self.clear_index()
        self.add_documents(documents)
        logger.info("Индекс перестроен")
    
    def save_index(self, path: str):
        """
        Сохраняет индекс в файл
        
        Args:
            path: Путь для сохранения
        """
        if self.index:
            faiss.write_index(self.index, path)
            logger.info(f"Индекс сохранен: {path}")
    
    def load_index(self, path: str):
        """
        Загружает индекс из файла
        
        Args:
            path: Путь к файлу индекса
        """
        try:
            self.index = faiss.read_index(path)
            logger.info(f"Индекс загружен: {path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")
            raise
