"""
Интеграционные тесты для RAG системы
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.medical_rag.core.rag_system import MedicalRAGSystem
from src.medical_rag.models.config import Config
from src.medical_rag.models.document import Document, DocumentMetadata


class TestRAGSystemIntegration:
    """Интеграционные тесты для RAG системы"""
    
    @pytest.fixture
    def sample_config(self):
        """Конфигурация для тестов"""
        config = Config()
        config.model.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        config.model.generation_model = "microsoft/DialoGPT-medium"
        config.retrieval.top_k = 3
        config.retrieval.similarity_threshold = 0.5
        config.debug = True
        return config
    
    @pytest.fixture
    def sample_documents(self):
        """Образцы документов для тестов"""
        metadata1 = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test1.docx"
        )
        
        metadata2 = DocumentMetadata(
            category="symptoms",
            difficulty="basic",
            topic="VBNK",
            source_file="test2.docx"
        )
        
        documents = [
            Document(
                id="test_001",
                content="Вопрос: Что такое варикозное расширение вен?\nОтвет: Варикозное расширение вен - это заболевание вен нижних конечностей.",
                question="Что такое варикозное расширение вен?",
                answer="Варикозное расширение вен - это заболевание вен нижних конечностей.",
                metadata=metadata1
            ),
            Document(
                id="test_002",
                content="Вопрос: Какие симптомы у варикоза?\nОтвет: Симптомы включают боль, тяжесть в ногах, отеки.",
                question="Какие симптомы у варикоза?",
                answer="Симптомы включают боль, тяжесть в ногах, отеки.",
                metadata=metadata2
            )
        ]
        
        return documents
    
    @pytest.fixture
    def sample_dataset(self, sample_documents):
        """Образец датасета для тестов"""
        return {
            "dataset_info": {
                "version": "1.0",
                "total_pairs": 2,
                "creation_date": "2024-01-01"
            },
            "pairs": [
                {
                    "id": doc.id,
                    "question": doc.question,
                    "answer": doc.answer,
                    "source": {
                        "file": doc.metadata.source_file,
                        "section": "test"
                    },
                    "metadata": {
                        "category": doc.metadata.category,
                        "difficulty": doc.metadata.difficulty,
                        "topic": doc.metadata.topic
                    }
                }
                for doc in sample_documents
            ],
            "validation_notes": ["Test dataset"]
        }
    
    def test_rag_system_initialization(self, sample_config):
        """Тест инициализации RAG системы"""
        rag = MedicalRAGSystem(sample_config)
        assert not rag.is_initialized()
        assert rag.config == sample_config
    
    def test_rag_system_with_mock_data(self, sample_config, sample_dataset, tmp_path):
        """Тест RAG системы с тестовыми данными"""
        # Создаем временный файл с данными
        dataset_path = tmp_path / "test_dataset.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dataset, f, ensure_ascii=False, indent=2)
        
        # Обновляем конфигурацию
        sample_config.data.dataset_path = str(dataset_path)
        
        # Инициализируем RAG систему
        rag = MedicalRAGSystem(sample_config)
        
        # Мокаем инициализацию сервисов для быстрого тестирования
        rag._initialize_services = lambda: None
        rag.embedding_service = None
        rag.retrieval_service = None
        rag.generation_service = None
        
        # Тестируем загрузку данных
        rag._load_and_process_data = lambda: None
        rag._initialized = True
        
        assert rag.is_initialized()
    
    def test_rag_system_info(self, sample_config):
        """Тест получения информации о системе"""
        rag = MedicalRAGSystem(sample_config)
        
        info = rag.get_system_info()
        assert info["status"] == "not_initialized"
        assert "config" in info
        assert "statistics" in info
        assert "services" in info
    
    def test_rag_system_save_load(self, sample_config, tmp_path):
        """Тест сохранения и загрузки системы"""
        rag = MedicalRAGSystem(sample_config)
        
        # Мокаем инициализацию
        rag._initialized = True
        rag.retrieval_service = type('MockRetrieval', (), {
            'documents': [],
            'get_statistics': lambda: {"total_documents": 0},
            'save_index': lambda path: None,
            'load_index': lambda path: None
        })()
        
        # Тестируем сохранение
        save_path = tmp_path / "test_system"
        rag.save_system(str(save_path))
        
        # Проверяем, что файлы созданы
        assert (save_path / "config.json").exists()
        assert (save_path / "documents.json").exists()
        assert (save_path / "vector_index.faiss").exists()
    
    def test_rag_system_error_handling(self, sample_config):
        """Тест обработки ошибок"""
        rag = MedicalRAGSystem(sample_config)
        
        # Тестируем обработку запроса без инициализации
        with pytest.raises(RuntimeError):
            rag.answer_question("Test question")
    
    def test_rag_system_config_validation(self):
        """Тест валидации конфигурации"""
        # Тест с некорректной конфигурацией
        config = Config()
        config.model.embedding_model = "invalid-model"
        
        rag = MedicalRAGSystem(config)
        assert rag.config.model.embedding_model == "invalid-model"
    
    def test_rag_system_logging(self, sample_config, caplog):
        """Тест логирования"""
        with caplog.at_level("INFO"):
            rag = MedicalRAGSystem(sample_config)
            assert "RAG система инициализирована" in caplog.text
    
    def test_rag_system_services_initialization(self, sample_config):
        """Тест инициализации сервисов"""
        rag = MedicalRAGSystem(sample_config)
        
        # Мокаем инициализацию сервисов
        rag._initialize_services()
        
        # Проверяем, что сервисы созданы
        assert rag.embedding_service is not None
        assert rag.retrieval_service is not None
        assert rag.generation_service is not None
        assert rag.data_processor is not None
    
    def test_rag_system_data_processing(self, sample_config, sample_dataset, tmp_path):
        """Тест обработки данных"""
        # Создаем временный файл с данными
        dataset_path = tmp_path / "test_dataset.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dataset, f, ensure_ascii=False, indent=2)
        
        sample_config.data.dataset_path = str(dataset_path)
        
        rag = MedicalRAGSystem(sample_config)
        
        # Тестируем создание документов из данных
        documents = rag._create_documents_from_data(sample_dataset)
        
        assert len(documents) == 2
        assert documents[0].id == "test_001"
        assert documents[1].id == "test_002"
        assert documents[0].question == "Что такое варикозное расширение вен?"
        assert documents[1].question == "Какие симптомы у варикоза?"
    
    def test_rag_system_document_creation_error_handling(self, sample_config):
        """Тест обработки ошибок при создании документов"""
        rag = MedicalRAGSystem(sample_config)
        
        # Тестовые данные с ошибками
        invalid_data = {
            "pairs": [
                {
                    "id": "test_001",
                    "question": "Test question?",
                    "answer": "Test answer",
                    # Отсутствуют метаданные
                },
                {
                    "id": "test_002",
                    # Отсутствует вопрос
                    "answer": "Test answer",
                    "metadata": {"category": "test"}
                }
            ]
        }
        
        # Тестируем создание документов с ошибками
        documents = rag._create_documents_from_data(invalid_data)
        
        # Должны быть созданы только валидные документы
        assert len(documents) == 0  # Все документы невалидны


class TestRAGSystemPerformance:
    """Тесты производительности RAG системы"""
    
    def test_rag_system_memory_usage(self, sample_config):
        """Тест использования памяти"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        rag = MedicalRAGSystem(sample_config)
        
        # Проверяем, что использование памяти разумное
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Использование памяти не должно превышать 100MB для базовой инициализации
        assert memory_increase < 100 * 1024 * 1024
    
    def test_rag_system_initialization_time(self, sample_config):
        """Тест времени инициализации"""
        import time
        
        start_time = time.time()
        rag = MedicalRAGSystem(sample_config)
        end_time = time.time()
        
        initialization_time = end_time - start_time
        
        # Инициализация должна быть быстрой (менее 1 секунды)
        assert initialization_time < 1.0
