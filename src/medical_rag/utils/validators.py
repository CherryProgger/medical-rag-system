"""
Валидаторы для медицинской RAG системы
"""

from typing import Any, Dict, List, Optional
from ..models.config import Config
from ..models.document import Document


def validate_config(config: Config) -> List[str]:
    """
    Валидирует конфигурацию системы
    
    Args:
        config: Конфигурация для валидации
        
    Returns:
        Список ошибок валидации
    """
    errors = []
    
    # Проверяем модели
    if not config.model.embedding_model:
        errors.append("Не указана модель для эмбеддингов")
    
    if not config.model.generation_model:
        errors.append("Не указана модель для генерации")
    
    # Проверяем параметры поиска
    if config.retrieval.top_k <= 0:
        errors.append("top_k должен быть положительным числом")
    
    if not 0 <= config.retrieval.similarity_threshold <= 1:
        errors.append("similarity_threshold должен быть между 0 и 1")
    
    # Проверяем параметры данных
    if not 0 <= config.data.test_size <= 1:
        errors.append("test_size должен быть между 0 и 1")
    
    if not 0 <= config.data.validation_size <= 1:
        errors.append("validation_size должен быть между 0 и 1")
    
    if config.data.test_size + config.data.validation_size > 1:
        errors.append("Сумма test_size и validation_size не должна превышать 1")
    
    # Проверяем системные параметры
    if config.max_workers <= 0:
        errors.append("max_workers должен быть положительным числом")
    
    return errors


def validate_document(document: Document) -> List[str]:
    """
    Валидирует документ
    
    Args:
        document: Документ для валидации
        
    Returns:
        Список ошибок валидации
    """
    errors = []
    
    # Проверяем обязательные поля
    if not document.id:
        errors.append("ID документа не может быть пустым")
    
    if not document.content:
        errors.append("Содержимое документа не может быть пустым")
    
    if not document.question:
        errors.append("Вопрос документа не может быть пустым")
    
    if not document.answer:
        errors.append("Ответ документа не может быть пустым")
    
    # Проверяем метаданные
    if document.metadata:
        if not document.metadata.category:
            errors.append("Категория документа не может быть пустой")
        
        if not document.metadata.topic:
            errors.append("Тема документа не может быть пустой")
        
        if document.metadata.difficulty not in ["basic", "intermediate", "advanced"]:
            errors.append("Сложность документа должна быть basic, intermediate или advanced")
    
    return errors


def validate_query_text(text: str) -> List[str]:
    """
    Валидирует текст запроса
    
    Args:
        text: Текст запроса
        
    Returns:
        Список ошибок валидации
    """
    errors = []
    
    if not text or not text.strip():
        errors.append("Текст запроса не может быть пустым")
    
    if len(text.strip()) < 3:
        errors.append("Текст запроса должен содержать минимум 3 символа")
    
    if len(text) > 1000:
        errors.append("Текст запроса не должен превышать 1000 символов")
    
    return errors


def validate_embedding(embedding: List[float]) -> List[str]:
    """
    Валидирует эмбеддинг
    
    Args:
        embedding: Эмбеддинг для валидации
        
    Returns:
        Список ошибок валидации
    """
    errors = []
    
    if not embedding:
        errors.append("Эмбеддинг не может быть пустым")
    
    if len(embedding) == 0:
        errors.append("Эмбеддинг должен содержать элементы")
    
    # Проверяем на NaN и бесконечность
    for i, value in enumerate(embedding):
        if not isinstance(value, (int, float)):
            errors.append(f"Элемент {i} эмбеддинга должен быть числом")
        elif not (float('-inf') < value < float('inf')):
            errors.append(f"Элемент {i} эмбеддинга содержит NaN или бесконечность")
    
    return errors


def validate_similarity_score(score: float) -> List[str]:
    """
    Валидирует оценку сходства
    
    Args:
        score: Оценка сходства
        
    Returns:
        Список ошибок валидации
    """
    errors = []
    
    if not isinstance(score, (int, float)):
        errors.append("Оценка сходства должна быть числом")
    elif not 0 <= score <= 1:
        errors.append("Оценка сходства должна быть между 0 и 1")
    elif not (float('-inf') < score < float('inf')):
        errors.append("Оценка сходства содержит NaN или бесконечность")
    
    return errors


def validate_dataset_path(path: str) -> List[str]:
    """
    Валидирует путь к датасету
    
    Args:
        path: Путь к датасету
        
    Returns:
        Список ошибок валидации
    """
    errors = []
    
    if not path:
        errors.append("Путь к датасету не может быть пустым")
    
    from pathlib import Path
    dataset_path = Path(path)
    
    if not dataset_path.exists():
        errors.append(f"Файл датасета не найден: {path}")
    
    if not dataset_path.suffix == '.json':
        errors.append("Файл датасета должен иметь расширение .json")
    
    return errors
