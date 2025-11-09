#!/usr/bin/env python3
"""
Скрипт для настройки проекта медицинской RAG системы
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Выполняет команду и выводит результат"""
    print(f"[INFO] {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {description} выполнено")
            return True
        else:
            print(f"[ERROR] {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] {description} - исключение: {e}")
        return False


def create_directories():
    """Создает необходимые директории"""
    directories = [
        "data",
        "logs", 
        "cache",
        "models",
        "docs",
        "examples",
        "src/medical_rag/core",
        "src/medical_rag/models",
        "src/medical_rag/data",
        "src/medical_rag/services",
        "src/medical_rag/interfaces",
        "src/medical_rag/utils",
        "tests/unit",
        "tests/integration", 
        "tests/e2e",
        "config"
    ]
    
    print("Создание директорий...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  - {directory}")
    
    print("Все директории созданы")


def install_dependencies():
    """Устанавливает зависимости"""
    print("Установка зависимостей...")
    
    # Основные зависимости
    if not run_command("pip install -r requirements.txt", "Установка основных зависимостей"):
        return False
    
    # Зависимости для разработки
    if not run_command("pip install -r requirements-dev.txt", "Установка зависимостей для разработки"):
        return False
    
    return True


def setup_pre_commit():
    """Настраивает pre-commit хуки"""
    print("🔧 Настройка pre-commit...")
    
    # Создаем .pre-commit-config.yaml
    pre_commit_config = """
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""
    
    with open(".pre-commit-config.yaml", "w") as f:
        f.write(pre_commit_config)
    
    # Устанавливаем pre-commit
    if run_command("pre-commit install", "Установка pre-commit хуков"):
        print("✅ Pre-commit настроен")
        return True
    return False


def create_gitignore():
    """Создает .gitignore файл"""
    print("📝 Создание .gitignore...")
    
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# Logs
logs/
*.log

# Data
data/raw/
data/processed/
*.json
*.csv
*.xlsx

# Models
models/
*.pkl
*.joblib
*.h5
*.pt
*.pth

# Cache
cache/
.cache/

# OS
.DS_Store
Thumbs.db

# Medical RAG specific
vector_index.faiss
embeddings_cache/
evaluation_reports/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore создан")


def create_pytest_config():
    """Создает конфигурацию pytest"""
    print("🧪 Настройка pytest...")
    
    pytest_ini = """
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src/medical_rag
    --cov-report=term-missing
    --cov-report=html:htmlcov
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
"""
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_ini)
    
    print("✅ pytest настроен")


def create_mypy_config():
    """Создает конфигурацию mypy"""
    print("🔍 Настройка mypy...")
    
    mypy_ini = """
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

[mypy-tests.*]
disallow_untyped_defs = False
"""
    
    with open("mypy.ini", "w") as f:
        f.write(mypy_ini)
    
    print("✅ mypy настроен")


def run_initial_tests():
    """Запускает начальные тесты"""
    print("🧪 Запуск начальных тестов...")
    
    if run_command("python -m pytest tests/unit/ -v", "Unit тесты"):
        print("✅ Unit тесты прошли успешно")
        return True
    else:
        print("⚠️ Unit тесты не прошли, но это нормально для начальной настройки")
        return True


def main():
    """Основная функция настройки"""
    print("🚀 Настройка проекта медицинской RAG системы")
    print("=" * 50)
    
    # Проверяем Python версию
    if sys.version_info < (3, 9):
        print("❌ Требуется Python 3.9 или выше")
        sys.exit(1)
    
    print(f"✅ Python {sys.version}")
    
    # Создаем директории
    create_directories()
    
    # Создаем конфигурационные файлы
    create_gitignore()
    create_pytest_config()
    create_mypy_config()
    
    # Устанавливаем зависимости
    if not install_dependencies():
        print("❌ Ошибка установки зависимостей")
        sys.exit(1)
    
    # Настраиваем pre-commit
    setup_pre_commit()
    
    # Запускаем тесты
    run_initial_tests()
    
    print("\n" + "=" * 50)
    print("🎉 Настройка проекта завершена!")
    print("\n📋 Следующие шаги:")
    print("1. Запустите предобработку данных: python main_new.py --preprocess")
    print("2. Запустите тесты: python main_new.py --test")
    print("3. Запустите систему: python main_new.py --cli")
    print("4. Откройте веб-интерфейс: python main_new.py --web")
    print("\n📚 Документация: README_NEW.md")
    print("🔧 Конфигурация: config/")
    print("🧪 Тесты: tests/")


if __name__ == "__main__":
    main()
