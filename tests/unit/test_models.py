"""
Юнит-тесты для моделей данных
"""

import pytest
from datetime import datetime
from src.medical_rag.models.document import Document, DocumentMetadata
from src.medical_rag.models.query import Query
from src.medical_rag.models.response import Response, ResponseMetadata


class TestDocumentMetadata:
    """Тесты для метаданных документа"""
    
    def test_metadata_creation(self):
        """Тест создания метаданных"""
        metadata = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test.docx"
        )
        
        assert metadata.category == "definition"
        assert metadata.difficulty == "basic"
        assert metadata.topic == "VBNK"
        assert metadata.source_file == "test.docx"
        assert metadata.confidence == "high"
    
    def test_metadata_to_dict(self):
        """Тест преобразования в словарь"""
        metadata = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test.docx"
        )
        
        data = metadata.to_dict()
        assert data["category"] == "definition"
        assert data["difficulty"] == "basic"
        assert data["topic"] == "VBNK"
        assert data["source_file"] == "test.docx"
    
    def test_metadata_from_dict(self):
        """Тест создания из словаря"""
        data = {
            "category": "definition",
            "difficulty": "basic",
            "topic": "VBNK",
            "source_file": "test.docx",
            "confidence": "high"
        }
        
        metadata = DocumentMetadata.from_dict(data)
        assert metadata.category == "definition"
        assert metadata.difficulty == "basic"
        assert metadata.topic == "VBNK"
        assert metadata.source_file == "test.docx"


class TestDocument:
    """Тесты для модели документа"""
    
    def test_document_creation(self):
        """Тест создания документа"""
        metadata = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test.docx"
        )
        
        document = Document(
            id="test_001",
            content="Test content",
            question="Test question?",
            answer="Test answer",
            metadata=metadata
        )
        
        assert document.id == "test_001"
        assert document.content == "Test content"
        assert document.question == "Test question?"
        assert document.answer == "Test answer"
        assert document.metadata.category == "definition"
    
    def test_document_validation(self):
        """Тест валидации документа"""
        metadata = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test.docx"
        )
        
        # Тест пустого ID
        with pytest.raises(ValueError):
            Document(
                id="",
                content="Test content",
                question="Test question?",
                answer="Test answer",
                metadata=metadata
            )
        
        # Тест пустого контента
        with pytest.raises(ValueError):
            Document(
                id="test_001",
                content="",
                question="Test question?",
                answer="Test answer",
                metadata=metadata
            )
    
    def test_document_to_dict(self):
        """Тест преобразования в словарь"""
        metadata = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test.docx"
        )
        
        document = Document(
            id="test_001",
            content="Test content",
            question="Test question?",
            answer="Test answer",
            metadata=metadata
        )
        
        data = document.to_dict()
        assert data["id"] == "test_001"
        assert data["content"] == "Test content"
        assert data["question"] == "Test question?"
        assert data["answer"] == "Test answer"
        assert data["metadata"]["category"] == "definition"
    
    def test_document_relevance(self):
        """Тест проверки релевантности"""
        metadata = DocumentMetadata(
            category="definition",
            difficulty="basic",
            topic="VBNK",
            source_file="test.docx"
        )
        
        document = Document(
            id="test_001",
            content="Test content",
            question="Test question?",
            answer="Test answer",
            metadata=metadata,
            similarity_score=0.7
        )
        
        assert document.is_relevant(0.5)
        assert not document.is_relevant(0.8)
        assert document.get_confidence_level() == "medium"


class TestQuery:
    """Тесты для модели запроса"""
    
    def test_query_creation(self):
        """Тест создания запроса"""
        query = Query(text="What is varicose veins?")
        
        assert query.text == "What is varicose veins?"
        assert query.max_results == 3
        assert query.similarity_threshold == 0.5
        assert query.timestamp is not None
    
    def test_query_validation(self):
        """Тест валидации запроса"""
        # Тест пустого текста
        with pytest.raises(ValueError):
            Query(text="")
        
        # Тест отрицательного max_results
        with pytest.raises(ValueError):
            Query(text="Test", max_results=-1)
        
        # Тест некорректного threshold
        with pytest.raises(ValueError):
            Query(text="Test", similarity_threshold=1.5)
    
    def test_query_medical_detection(self):
        """Тест определения медицинских запросов"""
        medical_query = Query(text="Что такое варикозное расширение вен?")
        non_medical_query = Query(text="Как приготовить борщ?")
        
        assert medical_query.is_medical_query()
        assert not non_medical_query.is_medical_query()
    
    def test_query_type_detection(self):
        """Тест определения типа запроса"""
        definition_query = Query(text="Что такое тромбоз?")
        procedure_query = Query(text="Как диагностировать варикоз?")
        list_query = Query(text="Какие симптомы у флебита?")
        
        assert definition_query.get_query_type() == "definition"
        assert procedure_query.get_query_type() == "procedure"
        assert list_query.get_query_type() == "list"


class TestResponse:
    """Тесты для модели ответа"""
    
    def test_response_creation(self):
        """Тест создания ответа"""
        metadata = ResponseMetadata(
            processing_time=1.5,
            num_documents_searched=100,
            num_documents_found=3,
            best_similarity_score=0.8,
            confidence_level="high",
            query_type="definition",
            is_medical_query=True
        )
        
        response = Response(
            query="What is varicose veins?",
            answer="Varicose veins are...",
            relevant_documents=[],
            metadata=metadata
        )
        
        assert response.query == "What is varicose veins?"
        assert response.answer == "Varicose veins are..."
        assert response.metadata.processing_time == 1.5
        assert response.metadata.confidence_level == "high"
    
    def test_response_validation(self):
        """Тест валидации ответа"""
        metadata = ResponseMetadata(
            processing_time=1.5,
            num_documents_searched=100,
            num_documents_found=3,
            best_similarity_score=0.8,
            confidence_level="high",
            query_type="definition",
            is_medical_query=True
        )
        
        # Тест пустого запроса
        with pytest.raises(ValueError):
            Response(
                query="",
                answer="Test answer",
                relevant_documents=[],
                metadata=metadata
            )
        
        # Тест пустого ответа
        with pytest.raises(ValueError):
            Response(
                query="Test query",
                answer="",
                relevant_documents=[],
                metadata=metadata
            )
    
    def test_response_confidence(self):
        """Тест проверки уверенности"""
        high_confidence_metadata = ResponseMetadata(
            processing_time=1.5,
            num_documents_searched=100,
            num_documents_found=3,
            best_similarity_score=0.8,
            confidence_level="high",
            query_type="definition",
            is_medical_query=True
        )
        
        low_confidence_metadata = ResponseMetadata(
            processing_time=1.5,
            num_documents_searched=100,
            num_documents_found=1,
            best_similarity_score=0.3,
            confidence_level="low",
            query_type="definition",
            is_medical_query=True
        )
        
        high_response = Response(
            query="Test",
            answer="Test answer",
            relevant_documents=[],
            metadata=high_confidence_metadata
        )
        
        low_response = Response(
            query="Test",
            answer="Test answer",
            relevant_documents=[],
            metadata=low_confidence_metadata
        )
        
        assert high_response.is_high_confidence()
        assert not low_response.is_high_confidence()
    
    def test_response_warnings(self):
        """Тест предупреждений"""
        metadata = ResponseMetadata(
            processing_time=1.5,
            num_documents_searched=100,
            num_documents_found=3,
            best_similarity_score=0.8,
            confidence_level="high",
            query_type="definition",
            is_medical_query=True
        )
        
        response = Response(
            query="Test",
            answer="Test answer",
            relevant_documents=[],
            metadata=metadata
        )
        
        assert not response.has_warnings()
        
        response.add_warning("Test warning")
        assert response.has_warnings()
        assert "Test warning" in response.warnings
