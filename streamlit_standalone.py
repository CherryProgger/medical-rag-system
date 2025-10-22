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
    generation_model: str = "microsoft/DialoGPT-medium"
    dataset_path: str = "data/rag_clean_dataset_v2_filtered.json"
    max_tokens: int = 256
    top_k: int = 3
    temperature: float = 0.7

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
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
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
            for item in data:
                doc = Document(
                    content=item.get('answer', ''),
                    metadata={'question': item.get('question', '')}
                )
                self.documents.append(doc)
            
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
            if not self.generation_model or not self.tokenizer:
                return "Модель генерации не загружена"
            
            # Создаем контекст из найденных документов
            context = "\n".join([doc.content for doc in context_docs[:2]])
            
            # Формируем промпт
            prompt = f"Контекст: {context}\n\nВопрос: {query}\n\nОтвет:"
            
            # Токенизируем
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодируем ответ
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ (после "Ответ:")
            if "Ответ:" in response:
                answer = response.split("Ответ:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()
            
            return answer if answer else "Не удалось сгенерировать ответ"
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return f"Ошибка генерации ответа: {str(e)}"
    
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
            return Config(**config_data)
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
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <p style='color: #2c3e50; font-size: 16px; line-height: 1.6;'>{response.answer}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if response.sources:
                    st.markdown("### Источники:")
                    for i, source in enumerate(response.sources, 1):
                        st.markdown(f"**{i}.** {source}")
                        
            except Exception as e:
                st.error(f"Ошибка при обработке вопроса: {str(e)}")
    
    st.markdown("---")
    st.markdown("### Примеры вопросов:")
    st.markdown("- Что такое варикозное расширение вен?")
    st.markdown("- Как лечить флебиты?")
    st.markdown("- Что такое тромбоэмболия?")
    st.markdown("- Какие симптомы ишемического инсульта?")

if __name__ == "__main__":
    main()
