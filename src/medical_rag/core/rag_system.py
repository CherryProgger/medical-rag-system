"""
Основная RAG система
"""

import time
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from ..models.config import Config
from ..models.document import Document
from ..models.query import Query
from ..models.response import Response
from .embedding_service import EmbeddingService
from .retrieval_service import RetrievalService
from .generation_service import GenerationService
from ..data.data_processor import DataProcessor

logger = logging.getLogger(__name__)


class MedicalRAGSystem:
    """
    Основная RAG система для медицинской документации
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Инициализация RAG системы
        
        Args:
            config: Конфигурация системы (если None, используется по умолчанию)
        """
        self.config = config or Config()
        self.embedding_service = None
        self.retrieval_service = None
        self.generation_service = None
        self.data_processor = None
        self._initialized = False
        
        # Настройка логирования
        self._setup_logging()
        
        logger.info("RAG система инициализирована")
    
    def _setup_logging(self):
        """Настраивает логирование"""
        logging.basicConfig(
            level=getattr(logging, self.config.logging.level),
            format=self.config.logging.format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.logs_dir / "rag_system.log") if self.config.logging.file_path else logging.NullHandler()
            ]
        )
    
    def initialize(self, dataset_path: Optional[str] = None) -> None:
        """
        Полная инициализация системы
        
        Args:
            dataset_path: Путь к датасету (если None, используется из конфигурации)
        """
        try:
            logger.info("Начинаем инициализацию RAG системы...")
            
            # Инициализация сервисов
            self._initialize_services()
            
            # Загрузка и обработка данных
            if dataset_path:
                self.config.data.dataset_path = dataset_path
            
            self._load_and_process_data()
            
            self._initialized = True
            logger.info("RAG система успешно инициализирована")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации RAG системы: {e}")
            raise
    
    def _initialize_services(self):
        """Инициализирует все сервисы"""
        logger.info("Инициализация сервисов...")
        
        # Сервис эмбеддингов
        self.embedding_service = EmbeddingService(self.config)
        
        # Сервис поиска
        self.retrieval_service = RetrievalService(self.config, self.embedding_service)
        
        # Сервис генерации
        self.generation_service = GenerationService(self.config)
        
        # Обработчик данных
        self.data_processor = DataProcessor(self.config)
        
        logger.info("Все сервисы инициализированы")
    
    def _load_and_process_data(self):
        """Загружает и обрабатывает данные"""
        logger.info("Загрузка и обработка данных...")
        
        # Загружаем обработанные данные
        processed_data = self.data_processor.load_processed_dataset()
        
        # Преобразуем в документы
        documents = self._create_documents_from_data(processed_data)
        
        # Добавляем документы в индекс
        self.retrieval_service.add_documents(documents)
        
        logger.info(f"Загружено {len(documents)} документов")
    
    def _create_documents_from_data(self, data: Dict[str, Any]) -> List[Document]:
        """Создает документы из данных"""
        documents = []
        
        for pair in data.get('pairs', []):
            try:
                # Создаем метаданные
                from ..models.document import DocumentMetadata
                metadata = DocumentMetadata(
                    category=pair.get('metadata', {}).get('category', ''),
                    difficulty=pair.get('metadata', {}).get('difficulty', ''),
                    topic=pair.get('metadata', {}).get('topic', ''),
                    source_file=pair.get('source', {}).get('file', ''),
                    section=pair.get('source', {}).get('section'),
                    confidence=pair.get('source', {}).get('confidence', 'high')
                )
                
                # Создаем документ
                document = Document(
                    id=pair['id'],
                    content=f"Вопрос: {pair['question']}\nОтвет: {pair['answer']}",
                    question=pair['question'],
                    answer=pair['answer'],
                    metadata=metadata
                )
                
                documents.append(document)
                
            except Exception as e:
                logger.warning(f"Ошибка создания документа {pair.get('id', 'unknown')}: {e}")
                continue
        
        return documents
    
    def answer_question(self, question_text: str, **kwargs) -> Response:
        """
        Основной метод для получения ответа на вопрос
        
        Args:
            question_text: Текст вопроса
            **kwargs: Дополнительные параметры для Query
            
        Returns:
            Ответ системы
        """
        if not self._initialized:
            raise RuntimeError("RAG система не инициализирована. Вызовите initialize() сначала.")
        
        start_time = time.time()
        
        try:
            # Создаем запрос
            query = Query(text=question_text, **kwargs)
            
            # Поиск релевантных документов
            relevant_documents = self.retrieval_service.search(query)
            
            # Генерация ответа
            processing_time = time.time() - start_time
            response = self.generation_service.generate_response(
                query, relevant_documents, processing_time
            )
            
            logger.info(f"Обработан запрос: '{question_text[:50]}...' за {processing_time:.2f}с")
            return response
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            # Возвращаем пустой ответ в случае ошибки
            processing_time = time.time() - start_time
            return Response(
                query=question_text,
                answer="Извините, произошла ошибка при обработке вашего запроса.",
                relevant_documents=[],
                metadata=ResponseMetadata(
                    processing_time=processing_time,
                    num_documents_searched=0,
                    num_documents_found=0,
                    best_similarity_score=0.0,
                    confidence_level="low",
                    query_type="error",
                    is_medical_query=False
                ),
                sources=[],
                warnings=[f"Ошибка обработки: {str(e)}"]
            )
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Возвращает информацию о системе
        
        Returns:
            Словарь с информацией о системе
        """
        if not self._initialized:
            return {"status": "not_initialized"}
        
        stats = self.retrieval_service.get_statistics()
        
        return {
            "status": "initialized",
            "config": self.config.to_dict(),
            "statistics": stats,
            "services": {
                "embedding_service": self.embedding_service.is_model_loaded(),
                "retrieval_service": self.retrieval_service.index is not None,
                "generation_service": self.generation_service.is_model_loaded()
            }
        }
    
    def save_system(self, save_path: str) -> None:
        """
        Сохраняет систему
        
        Args:
            save_path: Путь для сохранения
        """
        if not self._initialized:
            raise RuntimeError("Система не инициализирована")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем конфигурацию
        self.config.save(str(save_dir / "config.json"))
        
        # Сохраняем документы
        documents_data = [doc.to_dict() for doc in self.retrieval_service.documents]
        with open(save_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(documents_data, f, ensure_ascii=False, indent=2)
        
        # Сохраняем индекс
        self.retrieval_service.save_index(str(save_dir / "vector_index.faiss"))
        
        logger.info(f"Система сохранена в {save_path}")
    
    def load_system(self, load_path: str) -> None:
        """
        Загружает систему
        
        Args:
            load_path: Путь для загрузки
        """
        load_dir = Path(load_path)
        
        # Загружаем конфигурацию
        config_path = load_dir / "config.json"
        if config_path.exists():
            self.config = Config.load(str(config_path))
        
        # Инициализируем сервисы
        self._initialize_services()
        
        # Загружаем документы
        documents_path = load_dir / "documents.json"
        if documents_path.exists():
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            documents = [Document.from_dict(doc) for doc in documents_data]
            self.retrieval_service.add_documents(documents)
        
        # Загружаем индекс
        index_path = load_dir / "vector_index.faiss"
        if index_path.exists():
            self.retrieval_service.load_index(str(index_path))
        
        self._initialized = True
        logger.info(f"Система загружена из {load_path}")
    
    def is_initialized(self) -> bool:
        """Проверяет, инициализирована ли система"""
        return self._initialized
