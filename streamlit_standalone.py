#!/usr/bin/env python3
"""
Полностью автономный файл для Streamlit Cloud без зависимостей от модулей
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Документ для RAG системы"""
    content: str
    metadata: Dict[str, Any] = None

@dataclass
class Query:
    """Запрос пользователя"""
    text: str
    timestamp: str = None

@dataclass
class Response:
    """Ответ RAG системы"""
    answer: str
    sources: List[str] = None
    confidence: float = 0.0

@dataclass
class Config:
    """Конфигурация системы"""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    generation_model: str = "microsoft/DialoGPT-small"  # Более легкая модель
    dataset_path: str = "data/rag_clean_dataset_v2_filtered.json"
    max_tokens: int = 128  # Меньше токенов
    top_k: int = 3
    temperature: float = 0.7
    # Игнорируем неизвестные поля
    def __post_init__(self):
        pass

class SimpleRAGSystem:
    """Упрощенная RAG система без сложных импортов"""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def load_models(self):
        """Загружает модели"""
        try:
            logger.info("Загружаем модель эмбеддингов...")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            logger.info("Загружаем модель генерации...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.generation_model)
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                self.config.generation_model,
                torch_dtype=torch.float32
            )
            # Перемещаем модель на CPU
            self.generation_model = self.generation_model.to('cpu')
            
            # Добавляем pad_token если его нет
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Модели загружены успешно")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            return False
    
    def load_data(self):
        """Загружает данные"""
        try:
            data_path = Path(self.config.dataset_path)
            if not data_path.exists():
                logger.error(f"Файл данных не найден: {data_path}")
                return False
                
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.documents = []
            
            # Проверяем структуру данных
            if isinstance(data, dict):
                # Если это словарь, ищем ключ 'data' или 'questions'
                if 'data' in data:
                    items = data['data']
                elif 'questions' in data:
                    items = data['questions']
                else:
                    # Если нет стандартных ключей, берем все значения
                    items = list(data.values())
                    # Фильтруем только списки
                    items = [item for item in items if isinstance(item, list)]
                    if items:
                        items = items[0]  # Берем первый список
                    else:
                        logger.error("Не найдены данные в словаре")
                        return False
            elif isinstance(data, list):
                items = data
            else:
                logger.error(f"Неожиданная структура данных: {type(data)}")
                return False
            
            # Обрабатываем элементы
            for item in items:
                if isinstance(item, dict):
                    doc = Document(
                        content=item.get('answer', ''),
                        metadata={'question': item.get('question', '')}
                    )
                    self.documents.append(doc)
                else:
                    logger.warning(f"Неожиданный тип элемента: {type(item)}")
            
            logger.info(f"Загружено {len(self.documents)} документов")
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            return False
    
    def create_embeddings(self):
        """Создает эмбеддинги для документов"""
        try:
            if not self.embedding_model:
                logger.error("Модель эмбеддингов не загружена")
                return False
                
            texts = [doc.content for doc in self.documents]
            self.embeddings = self.embedding_model.encode(texts)
            
            # Создаем FAISS индекс
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            logger.info(f"Создано {len(self.documents)} эмбеддингов")
            return True
        except Exception as e:
            logger.error(f"Ошибка создания эмбеддингов: {e}")
            return False
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Document]:
        """Ищет похожие документы"""
        try:
            if not self.embedding_model or not self.index:
                return []
                
            query_embedding = self.embedding_model.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    results.append(self.documents[idx])
            
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Генерирует ответ на основе контекста"""
        try:
            # Простой fallback - возвращаем найденный документ
            if context_docs:
                # Берем первый найденный документ и форматируем как ответ
                best_doc = context_docs[0]
                answer = f"На основе медицинской документации:\n\n{best_doc.content}"
                
                # Ограничиваем длину
                if len(answer) > 800:
                    answer = answer[:800] + "..."
                
                return answer
            else:
                return "К сожалению, не удалось найти релевантную информацию для ответа на ваш вопрос."
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return "Произошла ошибка при генерации ответа. Попробуйте переформулировать вопрос."
    
    def query(self, question: str) -> Response:
        """Обрабатывает запрос пользователя"""
        try:
            # Ищем похожие документы
            similar_docs = self.search_similar(question, self.config.top_k)
            
            if not similar_docs:
                return Response(
                    answer="Извините, не удалось найти релевантную информацию для ответа на ваш вопрос.",
                    sources=[],
                    confidence=0.0
                )
            
            # Генерируем ответ
            answer = self.generate_answer(question, similar_docs)
            
            # Формируем источники
            sources = [doc.metadata.get('question', 'Неизвестный источник') for doc in similar_docs]
            
            return Response(
                answer=answer,
                sources=sources,
                confidence=0.8 if similar_docs else 0.0
            )
            
        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {e}")
            return Response(
                answer=f"Произошла ошибка при обработке запроса: {str(e)}",
                sources=[],
                confidence=0.0
            )

# Загружаем конфигурацию
def load_config():
    """Загружает конфигурацию"""
    try:
        config_path = Path("config/default.json")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Фильтруем только известные поля
            known_fields = {
                'embedding_model', 'generation_model', 'dataset_path', 
                'max_tokens', 'top_k', 'temperature'
            }
            filtered_data = {k: v for k, v in config_data.items() if k in known_fields}
            
            return Config(**filtered_data)
        else:
            return Config()
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return Config()

# Создаем RAG систему
@st.cache_resource
def load_rag_system():
    """Загружает RAG систему с кэшированием"""
    config = load_config()
    rag_system = SimpleRAGSystem(config)
    
    # Загружаем компоненты
    if not rag_system.load_models():
        st.error("Ошибка загрузки моделей")
        return None
    
    if not rag_system.load_data():
        st.error("Ошибка загрузки данных")
        return None
    
    if not rag_system.create_embeddings():
        st.error("Ошибка создания эмбеддингов")
        return None
    
    return rag_system

# Веб-интерфейс
def main():
    """Главная функция веб-интерфейса"""
    try:
        st.set_page_config(
            page_title="Medical RAG System",
            page_icon="🏥",
            layout="wide"
        )
        
        st.title("🏥 Medical RAG System")
        st.markdown("Система ответов на вопросы по медицинской документации")
        
        # Загружаем RAG систему
        with st.spinner("Загружаем систему..."):
            rag_system = load_rag_system()
        
        if rag_system is None:
            st.error("Не удалось загрузить RAG систему")
            return
        
        st.success("Система готова к работе!")
        
        # Форма для ввода вопроса
        with st.form("question_form"):
            question = st.text_area(
                "Введите ваш вопрос:",
                placeholder="Например: Что такое варикозное расширение вен?",
                height=100
            )
            submit_button = st.form_submit_button("Получить ответ")
        
        if submit_button and question:
            with st.spinner("Обрабатываем ваш вопрос..."):
                try:
                    response = rag_system.query(question)
                    
                    st.success("Ответ получен!")
                    st.markdown("### Ответ:")
                    
                    # Безопасное отображение ответа с правильными цветами
                    answer_text = str(response.answer) if response.answer else "Ответ не получен"
                    st.markdown(f"""
                    <div style='background-color: #2c3e50; padding: 20px; border-radius: 10px; margin: 10px 0; border: 2px solid #3498db;'>
                        <p style='color: #ecf0f1; font-size: 16px; line-height: 1.6; font-weight: 400;'>{answer_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if response.sources and len(response.sources) > 0:
                        st.markdown("### Источники:")
                        for i, source in enumerate(response.sources, 1):
                            source_text = str(source) if source else f"Источник {i}"
                            st.markdown(f"**{i}.** {source_text}")
                            
                except Exception as e:
                    st.error(f"Ошибка при обработке вопроса: {str(e)}")
        
        st.markdown("---")
        st.markdown("### Примеры вопросов:")
        st.markdown("- Что такое варикозное расширение вен?")
        st.markdown("- Как лечить флебиты?")
        st.markdown("- Что такое тромбоэмболия?")
        st.markdown("- Какие симптомы ишемического инсульта?")
        
    except Exception as e:
        st.error(f"Критическая ошибка приложения: {str(e)}")
        st.markdown("Попробуйте перезагрузить страницу.")

# Запуск приложения
if __name__ == "__main__":
    main()
