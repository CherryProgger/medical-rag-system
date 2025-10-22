#!/usr/bin/env python3
"""
Главный файл для Streamlit Cloud
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Импортируем веб-интерфейс
from src.medical_rag.interfaces.web_interface import create_web_app

# Создаем и запускаем приложение
app = create_web_app()
app()
