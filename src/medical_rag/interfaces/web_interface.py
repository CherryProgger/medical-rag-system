"""
Веб-интерфейс для медицинской RAG системы
"""

import streamlit as st
import sys
from pathlib import Path

# Добавляем путь к модулям
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from medical_rag.core.rag_system import MedicalRAGSystem
from medical_rag.models.config import Config


def create_web_app():
    """Создает веб-приложение"""
    # Настройка страницы
    st.set_page_config(
        page_title="Медицинская RAG система",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CSS стили
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .question-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .answer-box {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
            color: #2c3e50 !important;
        }
        .answer-box p {
            color: #2c3e50 !important;
            margin: 0.5rem 0;
            line-height: 1.6;
        }
        .source-box {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 5px;
            font-size: 0.9rem;
            color: #666;
        }
        .metric-box {
            background-color: #fff;
            padding: 0.5rem;
            border-radius: 5px;
            border: 1px solid #ddd;
            text-align: center;
        }
        /* Исправляем цвета для Streamlit */
        .stMarkdown {
            color: #2c3e50;
        }
        .stText {
            color: #2c3e50;
        }
        .stTextInput > div > div > input {
            color: #2c3e50;
        }
        .stTextArea > div > div > textarea {
            color: #2c3e50;
        }
        /* Улучшаем контрастность */
        .stApp {
            color: #2c3e50;
        }
        .stMarkdown p {
            color: #2c3e50 !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1f77b4 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    @st.cache_resource
    def load_rag_system():
        """Загружает RAG систему с кэшированием"""
        try:
            config = Config.load("config/default.json")
            rag = MedicalRAGSystem(config)
            rag.initialize("data/rag_clean_dataset_v2_filtered.json")
            return rag
        except Exception as e:
            st.error(f"Ошибка загрузки RAG системы: {e}")
            return None
    
    def display_sidebar():
        """Отображает боковую панель с информацией"""
        st.sidebar.title("🏥 Медицинская RAG система")
        
        st.sidebar.markdown("### О системе")
        st.sidebar.info("""
        Эта система использует технологию RAG (Retrieval-Augmented Generation) 
        для ответов на вопросы по медицинской документации.
        
        **Возможности:**
        - Поиск по медицинским документам
        - Ответы на вопросы о сосудистых заболеваниях
        - Ссылки на источники информации
        """)
        
        st.sidebar.markdown("### Статистика")
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            rag = st.session_state.rag_system
            info = rag.get_system_info()
            if 'statistics' in info:
                st.sidebar.metric("Документов в базе", info['statistics'].get('total_documents', 0))
        
        st.sidebar.markdown("### Настройки")
        top_k = st.sidebar.slider("Количество документов для поиска", 1, 10, 3)
        
        return top_k
    
    def display_main_interface(rag_system, top_k):
        """Отображает основной интерфейс"""
        st.markdown('<h1 class="main-header">🏥 Медицинская RAG система</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Задайте вопрос по медицинской документации
        
        Система поможет найти ответы на вопросы о:
        - Варикозном расширении вен
        - Тромбоэмболических осложнениях  
        - Флебитах и тромбофлебитах
        - Диагностике и лечении сосудистых заболеваний
        """)
        
        # Поле для ввода вопроса
        question = st.text_area(
            "Введите ваш вопрос:",
            placeholder="Например: Что такое варикозное расширение вен?",
            height=100
        )
        
        # Кнопка поиска
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            search_button = st.button("🔍 Найти ответ", type="primary", use_container_width=True)
        
        # Обработка запроса
        if search_button and question:
            with st.spinner("Поиск ответа..."):
                response = rag_system.answer_question(question, max_results=top_k)
            
            # Отображение результата
            st.markdown("### 📋 Ответ")
            
            # Форматируем ответ для лучшей читаемости
            formatted_answer = response.answer.replace('\n', '<br>')
            st.markdown(f'<div class="answer-box"><p>{formatted_answer}</p></div>', unsafe_allow_html=True)
            
            # Метрики
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Время поиска", f"{response.metadata.processing_time:.2f}с")
            with col2:
                st.metric("Найдено документов", response.metadata.num_documents_found)
            with col3:
                st.metric("Релевантность", f"{response.metadata.best_similarity_score:.3f}")
            with col4:
                confidence = "Высокая" if response.metadata.confidence_level == "high" else "Средняя" if response.metadata.confidence_level == "medium" else "Низкая"
                st.metric("Уверенность", confidence)
            
            # Источники
            if response.relevant_documents:
                st.markdown("### 📚 Источники")
                for i, doc in enumerate(response.relevant_documents, 1):
                    with st.expander(f"Источник {i} (релевантность: {doc.similarity_score:.3f})"):
                        st.markdown(f"**Вопрос:** {doc.question}")
                        st.markdown(f"**Ответ:** {doc.answer}")
                        
                        if doc.metadata:
                            st.markdown(f"**Файл:** {doc.metadata.source_file}")
                            if doc.metadata.section:
                                st.markdown(f"**Раздел:** {doc.metadata.section}")
                            
                            st.markdown(f"**Категория:** {doc.metadata.category}")
                            st.markdown(f"**Сложность:** {doc.metadata.difficulty}")
                            st.markdown(f"**Тема:** {doc.metadata.topic}")
            
            # Предупреждения
            if response.warnings:
                st.markdown("### ⚠️ Предупреждения")
                for warning in response.warnings:
                    st.warning(warning)
    
    def display_example_questions():
        """Отображает примеры вопросов"""
        st.markdown("### 💡 Примеры вопросов")
        
        example_questions = [
            "Что такое варикозное расширение вен нижних конечностей?",
            "Какие симптомы характерны для тромбоза глубоких вен?",
            "Как диагностировать хронические заболевания вен?",
            "Какая классификация используется для заболеваний вен?",
            "Какие факторы риска у тромбофлебита поверхностных вен?",
            "Как лечить венозные тромбоэмболические осложнения?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}", use_container_width=True):
                    st.session_state.example_question = question
                    st.rerun()
    
    def main():
        """Основная функция приложения"""
        # Загрузка RAG системы
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = load_rag_system()
        
        if st.session_state.rag_system is None:
            st.error("Не удалось загрузить RAG систему. Проверьте наличие файлов данных.")
            return
        
        # Отображение интерфейса
        top_k = display_sidebar()
        display_example_questions()
        
        # Обработка примеров вопросов
        if 'example_question' in st.session_state:
            st.text_area("Введите ваш вопрос:", value=st.session_state.example_question, height=100)
            del st.session_state.example_question
        
        display_main_interface(st.session_state.rag_system, top_k)
        
        # Футер
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            Медицинская RAG система | Профессиональная версия | 
            <a href="https://github.com" target="_blank">GitHub</a>
        </div>
        """, unsafe_allow_html=True)
    
    return main


if __name__ == "__main__":
    app = create_web_app()
    app()
