"""
Обработчик данных для RAG системы
"""

import json
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from ..models.config import Config

logger = logging.getLogger(__name__)


class DataProcessor:
    """Класс для обработки данных RAG системы"""
    
    def __init__(self, config: Config):
        """
        Инициализация обработчика данных
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Загружает исходный датасет
        
        Args:
            dataset_path: Путь к исходному датасету
            
        Returns:
            Загруженные данные
        """
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            logger.info(f"Загружен датасет: {self.raw_data.get('dataset_info', {}).get('total_pairs', 0)} пар")
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            raise
    
    def filter_excluded_questions(self, excluded_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        """
        Исключает вопросы в указанном диапазоне
        
        Args:
            excluded_range: Диапазон вопросов для исключения (start, end)
            
        Returns:
            Отфильтрованные пары вопрос-ответ
        """
        if not self.raw_data:
            raise ValueError("Сначала загрузите исходный датасет")
        
        if excluded_range is None:
            excluded_range = self.config.data.excluded_range
        
        all_pairs = self.raw_data['pairs']
        start_idx, end_idx = excluded_range
        
        # Фильтруем пары, исключая указанный диапазон
        filtered_pairs = []
        excluded_count = 0
        
        for i, pair in enumerate(all_pairs):
            # Проверяем, попадает ли пара в исключаемый диапазон
            if start_idx <= i + 1 <= end_idx:  # +1 потому что индексация с 1
                excluded_count += 1
                logger.debug(f"Исключена пара {i+1}: {pair['question'][:50]}...")
            else:
                filtered_pairs.append(pair)
        
        logger.info(f"Исключено {excluded_count} пар из диапазона {start_idx}-{end_idx}")
        logger.info(f"Осталось {len(filtered_pairs)} пар для обучения")
        
        return filtered_pairs
    
    def create_processed_dataset(self, excluded_range: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Создает обработанный датасет без исключенных вопросов
        
        Args:
            excluded_range: Диапазон вопросов для исключения
            
        Returns:
            Обработанный датасет
        """
        if excluded_range is None:
            excluded_range = self.config.data.excluded_range
        
        filtered_pairs = self.filter_excluded_questions(excluded_range)
        
        processed_data = {
            "dataset_info": {
                **self.raw_data['dataset_info'],
                "total_pairs": len(filtered_pairs),
                "excluded_range": f"{excluded_range[0]}-{excluded_range[1]}",
                "excluded_count": self.raw_data['dataset_info']['total_pairs'] - len(filtered_pairs),
                "processing_notes": "Исключены вопросы 30-60 согласно комментарию о возможных глюках"
            },
            "pairs": filtered_pairs,
            "validation_notes": [
                *self.raw_data['validation_notes'],
                f"Исключены вопросы {excluded_range[0]}-{excluded_range[1]} для избежания глюков",
                f"Итого пар для обучения: {len(filtered_pairs)}"
            ]
        }
        
        self.processed_data = processed_data
        return processed_data
    
    def save_processed_dataset(self, output_path: str = None) -> str:
        """
        Сохраняет обработанный датасет
        
        Args:
            output_path: Путь для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        if not self.processed_data:
            self.create_processed_dataset()
        
        if not output_path:
            base_name = Path(self.config.data.dataset_path).stem
            output_path = f"{base_name}_filtered.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Обработанный датасет сохранен: {output_path}")
        return output_path
    
    def load_processed_dataset(self, dataset_path: str = None) -> Dict[str, Any]:
        """
        Загружает обработанный датасет
        
        Args:
            dataset_path: Путь к обработанному датасету
            
        Returns:
            Обработанные данные
        """
        if dataset_path is None:
            dataset_path = self.config.data.dataset_path
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
            
            logger.info(f"Загружен обработанный датасет: {self.processed_data.get('dataset_info', {}).get('total_pairs', 0)} пар")
            return self.processed_data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки обработанного датасета: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Возвращает статистику по обработанному датасету
        
        Returns:
            Словарь со статистикой
        """
        if not self.processed_data:
            return {}
        
        pairs = self.processed_data['pairs']
        
        # Статистика по категориям
        categories = {}
        topics = {}
        difficulties = {}
        
        for pair in pairs:
            category = pair.get('metadata', {}).get('category', 'unknown')
            topic = pair.get('metadata', {}).get('topic', 'unknown')
            difficulty = pair.get('metadata', {}).get('difficulty', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            topics[topic] = topics.get(topic, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        return {
            "total_pairs": len(pairs),
            "categories": categories,
            "topics": topics,
            "difficulties": difficulties,
            "excluded_count": self.raw_data['dataset_info']['total_pairs'] - len(pairs) if self.raw_data else 0
        }
    
    def split_dataset(self, test_size: float = None, validation_size: float = None, 
                     random_seed: int = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Разделяет датасет на обучающую, валидационную и тестовую выборки
        
        Args:
            test_size: Размер тестовой выборки
            validation_size: Размер валидационной выборки
            random_seed: Случайное зерно
            
        Returns:
            Кортеж (train, validation, test)
        """
        if not self.processed_data:
            raise ValueError("Сначала загрузите обработанный датасет")
        
        if test_size is None:
            test_size = self.config.data.test_size
        if validation_size is None:
            validation_size = self.config.data.validation_size
        if random_seed is None:
            random_seed = self.config.data.random_seed
        
        pairs = self.processed_data['pairs']
        
        # Устанавливаем случайное зерно
        np.random.seed(random_seed)
        
        # Сначала разделяем на train+val и test
        train_val, test = train_test_split(
            pairs, 
            test_size=test_size, 
            random_state=random_seed
        )
        
        # Затем разделяем train+val на train и val
        if validation_size > 0:
            val_size = validation_size / (1 - test_size)  # Нормализуем размер валидации
            train, validation = train_test_split(
                train_val,
                test_size=val_size,
                random_state=random_seed
            )
        else:
            train = train_val
            validation = []
        
        logger.info(f"Разделение датасета: train={len(train)}, validation={len(validation)}, test={len(test)}")
        
        return train, validation, test
    
    def create_evaluation_dataset(self, test_size: float = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Создает датасет для оценки системы
        
        Args:
            test_size: Размер тестовой выборки
            
        Returns:
            Кортеж (тестовые вопросы, правильные ответы)
        """
        if test_size is None:
            test_size = self.config.data.test_size
        
        _, _, test_pairs = self.split_dataset(test_size=test_size)
        
        test_questions = [pair['question'] for pair in test_pairs]
        ground_truth = []
        
        for pair in test_pairs:
            gt = {
                'answer': pair['answer'],
                'relevant_doc_ids': [pair['id']],
                'source': pair.get('source', {}),
                'metadata': pair.get('metadata', {})
            }
            ground_truth.append(gt)
        
        logger.info(f"Создан датасет для оценки: {len(test_questions)} вопросов")
        return test_questions, ground_truth
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Валидирует качество данных
        
        Returns:
            Словарь с результатами валидации
        """
        if not self.processed_data:
            return {"error": "Нет обработанных данных"}
        
        pairs = self.processed_data['pairs']
        issues = []
        
        # Проверяем наличие обязательных полей
        for i, pair in enumerate(pairs):
            if not pair.get('id'):
                issues.append(f"Пара {i}: отсутствует ID")
            if not pair.get('question'):
                issues.append(f"Пара {i}: отсутствует вопрос")
            if not pair.get('answer'):
                issues.append(f"Пара {i}: отсутствует ответ")
            if not pair.get('metadata'):
                issues.append(f"Пара {i}: отсутствуют метаданные")
        
        # Проверяем дубликаты
        ids = [pair.get('id') for pair in pairs if pair.get('id')]
        duplicates = len(ids) - len(set(ids))
        
        return {
            "total_pairs": len(pairs),
            "issues": issues,
            "duplicates": duplicates,
            "quality_score": max(0, 1 - len(issues) / len(pairs) - duplicates / len(pairs))
        }
