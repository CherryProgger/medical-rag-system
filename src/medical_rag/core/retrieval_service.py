"""
Сервис для поиска релевантных документов
"""

import logging
from typing import List, Optional, Tuple

import faiss
import numpy as np

from ..cache.embedding_cache import EmbeddingCache
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
        self.embedding_cache: Optional[EmbeddingCache] = None
        self._initialize_index()
        if self.config.cache_embeddings:
            try:
                self.embedding_cache = EmbeddingCache(self.config.embeddings_cache_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Не удалось инициализировать кэш эмбеддингов: %s", exc)
    
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
            prepared_documents = self._ensure_document_embeddings(documents)
            embeddings = [doc.embedding for doc in prepared_documents if doc.embedding is not None]
            if not embeddings:
                logger.warning("Нет валидных эмбеддингов для добавления")
                return
            
            embeddings_array = np.array(embeddings, dtype=np.float32)
            embeddings_normalized = self.embedding_service.normalize_embeddings(embeddings_array)
            self.index.add(embeddings_normalized)
            self.documents.extend(prepared_documents)
            logger.info(f"Добавлено {len(embeddings)} документов в индекс")
            
        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")
            raise
    
    def _ensure_document_embeddings(self, documents: List[Document]) -> List[Document]:
        """
        Гарантирует наличие эмбеддингов у документов (использует кэш при необходимости).
        """
        documents_to_encode: List[Document] = []
        cache_keys: List[Optional[str]] = []
        
        for doc in documents:
            cache_key = doc.get_cache_key()
            cached_embedding = None
            if cache_key and self.embedding_cache:
                cached_embedding = self.embedding_cache.get(cache_key)
                if cached_embedding is not None:
                    doc.embedding = cached_embedding
            if doc.embedding is None:
                documents_to_encode.append(doc)
                cache_keys.append(cache_key)
        
        if documents_to_encode:
            texts = [doc.content for doc in documents_to_encode]
            new_embeddings = self.embedding_service.create_embeddings(texts)
            for doc, embedding, cache_key in zip(documents_to_encode, new_embeddings, cache_keys):
                embedding_list = embedding.astype(np.float32).tolist()
                doc.embedding = embedding_list
                if cache_key and self.embedding_cache:
                    self.embedding_cache.set(cache_key, embedding_list)
        
        return documents
    
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
