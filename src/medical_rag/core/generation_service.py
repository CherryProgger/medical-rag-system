"""
Сервис для генерации ответов
"""

import torch
from typing import List, Optional, Dict, Any
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ..models.config import Config
from ..models.document import Document
from ..models.query import Query
from ..models.response import Response, ResponseMetadata

logger = logging.getLogger(__name__)


class GenerationService:
    """Сервис для генерации ответов на основе найденных документов"""
    
    def __init__(self, config: Config):
        """
        Инициализация сервиса генерации
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generator = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализирует модель генерации"""
        try:
            logger.info(f"Загружаем модель генерации: {self.config.model.generation_model}")
            
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.generation_model)
            
            # Загружаем модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.generation_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Настраиваем токенизатор
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Создаем pipeline для генерации
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Модель генерации успешно загружена")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели генерации: {e}")
            logger.info("Используем простую генерацию на основе найденных документов")
            self.model = None
            self.tokenizer = None
            self.generator = None
    
    def generate_answer(self, query: Query, relevant_documents: List[Document]) -> str:
        """
        Генерирует ответ на основе запроса и релевантных документов
        
        Args:
            query: Запрос пользователя
            relevant_documents: Релевантные документы
            
        Returns:
            Сгенерированный ответ
        """
        if not relevant_documents:
            return "Извините, не удалось найти релевантную информацию для ответа на ваш вопрос."
        
        try:
            # Если есть модель генерации, используем её
            if self.generator:
                return self._generate_with_model(query, relevant_documents)
            else:
                # Простая генерация на основе найденных документов
                return self._generate_simple_answer(query, relevant_documents)
                
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return self._generate_simple_answer(query, relevant_documents)
    
    def _generate_simple_answer(self, query: Query, relevant_documents: List[Document]) -> str:
        """
        Простая генерация ответа на основе наиболее релевантного документа
        
        Args:
            query: Запрос пользователя
            relevant_documents: Релевантные документы
            
        Returns:
            Ответ
        """
        # Находим наиболее релевантный документ
        best_doc = max(relevant_documents, key=lambda x: x.similarity_score or 0)
        
        # Возвращаем ответ из наиболее релевантного документа
        answer = best_doc.answer
        
        # Добавляем информацию об источнике
        source_info = best_doc.get_source_info()
        answer += f"\n\nИсточник: {source_info}"
        
        return answer
    
    def _generate_with_model(self, query: Query, relevant_documents: List[Document]) -> str:
        """
        Генерация ответа с использованием языковой модели
        
        Args:
            query: Запрос пользователя
            relevant_documents: Релевантные документы
            
        Returns:
            Сгенерированный ответ
        """
        try:
            # Формируем контекст из релевантных документов
            context = self._build_context(relevant_documents[:2])  # Берем топ-2 документа
            
            # Формируем промпт
            prompt = self._build_prompt(query.text, context)
            
            # Генерируем ответ
            generated_text = self.generator(
                prompt,
                max_length=len(prompt.split()) + 100,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                do_sample=self.config.model.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )[0]['generated_text']
            
            # Извлекаем только ответ
            answer = self._extract_answer(generated_text, prompt)
            
            return answer if answer else self._generate_simple_answer(query, relevant_documents)
            
        except Exception as e:
            logger.error(f"Ошибка генерации с моделью: {e}")
            return self._generate_simple_answer(query, relevant_documents)
    
    def _build_context(self, documents: List[Document]) -> str:
        """
        Строит контекст из документов
        
        Args:
            documents: Список документов
            
        Returns:
            Контекст
        """
        context_parts = []
        for doc in documents:
            context_parts.append(f"Вопрос: {doc.question}\nОтвет: {doc.answer}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Строит промпт для генерации
        
        Args:
            query: Запрос пользователя
            context: Контекст из документов
            
        Returns:
            Промпт
        """
        return f"""Контекст: {context}

Вопрос: {query}

Ответ:"""
    
    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """
        Извлекает ответ из сгенерированного текста
        
        Args:
            generated_text: Сгенерированный текст
            prompt: Исходный промпт
            
        Returns:
            Извлеченный ответ
        """
        # Удаляем промпт из сгенерированного текста
        if prompt in generated_text:
            answer = generated_text.replace(prompt, "").strip()
        else:
            # Если промпт не найден, ищем "Ответ:"
            answer_start = generated_text.find("Ответ:")
            if answer_start != -1:
                answer = generated_text[answer_start + len("Ответ:"):].strip()
            else:
                answer = generated_text.strip()
        
        return answer
    
    def generate_response(self, query: Query, relevant_documents: List[Document], 
                         processing_time: float) -> Response:
        """
        Генерирует полный ответ с метаданными
        
        Args:
            query: Запрос пользователя
            relevant_documents: Релевантные документы
            processing_time: Время обработки
            
        Returns:
            Полный ответ
        """
        # Генерируем текст ответа
        answer = self.generate_answer(query, relevant_documents)
        
        # Создаем метаданные
        metadata = ResponseMetadata(
            processing_time=processing_time,
            num_documents_searched=len(self._get_all_documents()) if hasattr(self, '_get_all_documents') else 0,
            num_documents_found=len(relevant_documents),
            best_similarity_score=max([doc.similarity_score or 0 for doc in relevant_documents]) if relevant_documents else 0.0,
            confidence_level=self._get_confidence_level(relevant_documents),
            query_type=query.get_query_type(),
            is_medical_query=query.is_medical_query()
        )
        
        # Создаем источники
        sources = [doc.get_source_info() for doc in relevant_documents]
        
        # Создаем предупреждения
        warnings = self._generate_warnings(relevant_documents, metadata)
        
        return Response(
            query=query.text,
            answer=answer,
            relevant_documents=relevant_documents,
            metadata=metadata,
            sources=sources,
            warnings=warnings
        )
    
    def _get_confidence_level(self, relevant_documents: List[Document]) -> str:
        """
        Определяет уровень уверенности
        
        Args:
            relevant_documents: Релевантные документы
            
        Returns:
            Уровень уверенности
        """
        if not relevant_documents:
            return "low"
        
        best_score = max([doc.similarity_score or 0 for doc in relevant_documents])
        
        if best_score >= 0.8:
            return "high"
        elif best_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_warnings(self, relevant_documents: List[Document], metadata: ResponseMetadata) -> List[str]:
        """
        Генерирует предупреждения
        
        Args:
            relevant_documents: Релевантные документы
            metadata: Метаданные ответа
            
        Returns:
            Список предупреждений
        """
        warnings = []
        
        if metadata.best_similarity_score < 0.5:
            warnings.append("Низкая релевантность найденных документов")
        
        if metadata.num_documents_found == 0:
            warnings.append("Не найдено релевантных документов")
        
        if metadata.processing_time > 5.0:
            warnings.append("Медленная обработка запроса")
        
        return warnings
    
    def is_model_loaded(self) -> bool:
        """Проверяет, загружена ли модель генерации"""
        return self.generator is not None
