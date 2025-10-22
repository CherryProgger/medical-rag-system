#!/usr/bin/env python3
"""
Главный файл для Streamlit Cloud
"""

import sys
import os
from pathlib import Path

# Добавляем путь к модулям
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Устанавливаем PYTHONPATH для Streamlit Cloud
os.environ['PYTHONPATH'] = f"{project_root}:{project_root / 'src'}"

# Импортируем веб-интерфейс напрямую
try:
    from medical_rag.interfaces.web_interface import create_web_app
    app = create_web_app()
    app()
except ImportError:
    # Fallback - импортируем напрямую
    import streamlit as st
    from medical_rag.core.rag_system import MedicalRAGSystem
    from medical_rag.data.data_processor import DataProcessor
    from medical_rag.models.config import Config
    import json
    
    # Загружаем конфигурацию
    config_path = project_root / "config" / "default.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    config = Config(**config_data)
    
    # Создаем RAG систему
    @st.cache_resource
    def load_rag_system():
        rag_system = MedicalRAGSystem(config)
        return rag_system
    
    # Веб-интерфейс
    st.set_page_config(
        page_title="Medical RAG System",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Medical RAG System")
    st.markdown("Система ответов на вопросы по медицинской документации")
    
    # Загружаем RAG систему
    rag_system = load_rag_system()
    
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
                st.markdown(f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;'><p style='color: #2c3e50; font-size: 16px; line-height: 1.6;'>{response.answer}</p></div>", unsafe_allow_html=True)
                
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
