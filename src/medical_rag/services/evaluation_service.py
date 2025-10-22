"""
Сервис оценки качества RAG системы
"""

import json
import numpy as np
import time
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

from ..core.rag_system import MedicalRAGSystem
from ..models.query import Query
from ..models.response import Response

logger = logging.getLogger(__name__)


class EvaluationService:
    """Сервис для оценки качества RAG системы"""
    
    def __init__(self, rag_system: MedicalRAGSystem):
        """
        Инициализация сервиса оценки
        
        Args:
            rag_system: RAG система для тестирования
        """
        self.rag_system = rag_system
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def evaluate_retrieval_quality(self, test_questions: List[str], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Оценивает качество поиска релевантных документов"""
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for question, gt in zip(test_questions, ground_truth):
            # Получаем релевантные документы
            query = Query(text=question, max_results=5)
            relevant_docs = self.rag_system.retrieval_service.search(query)
            
            # Извлекаем ID найденных документов
            retrieved_ids = [doc.id for doc in relevant_docs]
            
            # ID правильных документов
            correct_ids = set(gt.get('relevant_doc_ids', []))
            retrieved_ids_set = set(retrieved_ids)
            
            # Вычисляем метрики
            if len(retrieved_ids_set) > 0:
                precision = len(correct_ids.intersection(retrieved_ids_set)) / len(retrieved_ids_set)
            else:
                precision = 0.0
                
            if len(correct_ids) > 0:
                recall = len(correct_ids.intersection(retrieved_ids_set)) / len(correct_ids)
            else:
                recall = 0.0
                
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
                
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        return {
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1_score': np.mean(f1_scores),
            'precision_std': np.std(precision_scores),
            'recall_std': np.std(recall_scores),
            'f1_std': np.std(f1_scores)
        }
    
    def evaluate_answer_quality(self, test_questions: List[str], ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Оценивает качество генерируемых ответов"""
        semantic_similarities = []
        keyword_overlaps = []
        answer_lengths = []
        
        for question, gt in zip(test_questions, ground_truth):
            # Получаем ответ от RAG системы
            response = self.rag_system.answer_question(question)
            generated_answer = response.answer
            correct_answer = gt['answer']
            
            # Семантическое сходство
            semantic_sim = self._calculate_semantic_similarity(generated_answer, correct_answer)
            semantic_similarities.append(semantic_sim)
            
            # Пересечение ключевых слов
            keyword_overlap = self._calculate_keyword_overlap(generated_answer, correct_answer)
            keyword_overlaps.append(keyword_overlap)
            
            # Длина ответа
            answer_lengths.append(len(generated_answer.split()))
        
        return {
            'semantic_similarity': np.mean(semantic_similarities),
            'keyword_overlap': np.mean(keyword_overlaps),
            'avg_answer_length': np.mean(answer_lengths),
            'semantic_similarity_std': np.std(semantic_similarities),
            'keyword_overlap_std': np.std(keyword_overlaps),
            'answer_length_std': np.std(answer_lengths)
        }
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет семантическое сходство между двумя текстами"""
        try:
            # Используем эмбеддинги из RAG системы
            emb1 = self.rag_system.embedding_service.create_embedding(text1)
            emb2 = self.rag_system.embedding_service.create_embedding(text2)
            
            similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
            return float(similarity)
        except Exception:
            # Fallback к TF-IDF
            return self._calculate_tfidf_similarity(text1, text2)
    
    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Вычисляет сходство на основе TF-IDF"""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """Вычисляет пересечение ключевых слов"""
        import re
        # Извлекаем слова (убираем пунктуацию)
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_response_time(self, test_questions: List[str], num_runs: int = 3) -> Dict[str, float]:
        """Оценивает время отклика системы"""
        response_times = []
        
        for _ in range(num_runs):
            for question in test_questions:
                start_time = time.time()
                self.rag_system.answer_question(question)
                end_time = time.time()
                response_times.append(end_time - start_time)
        
        return {
            'avg_response_time': np.mean(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'std_response_time': np.std(response_times),
            'median_response_time': np.median(response_times)
        }
    
    def create_test_dataset(self, dataset_path: str, test_size: float = 0.2) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Создает тестовый датасет из исходного"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pairs = data['pairs']
        np.random.seed(42)
        np.random.shuffle(pairs)
        
        test_count = int(len(pairs) * test_size)
        test_pairs = pairs[:test_count]
        
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
        
        return test_questions, ground_truth
    
    def run_full_evaluation(self, dataset_path: str = None) -> Dict[str, Any]:
        """Запускает полную оценку системы"""
        if dataset_path is None:
            dataset_path = "data/rag_clean_dataset_v2_filtered.json"
        
        print("Создание тестового датасета...")
        test_questions, ground_truth = self.create_test_dataset(dataset_path, test_size=0.3)
        
        print(f"Тестирование на {len(test_questions)} вопросах...")
        
        # Оценка качества поиска
        print("Оценка качества поиска...")
        retrieval_metrics = self.evaluate_retrieval_quality(test_questions, ground_truth)
        
        # Оценка качества ответов
        print("Оценка качества ответов...")
        answer_metrics = self.evaluate_answer_quality(test_questions, ground_truth)
        
        # Оценка времени отклика
        print("Оценка времени отклика...")
        time_metrics = self.evaluate_response_time(test_questions)
        
        # Общая оценка
        overall_score = (
            retrieval_metrics['f1_score'] * 0.4 +
            answer_metrics['semantic_similarity'] * 0.4 +
            (1 - min(time_metrics['avg_response_time'] / 5, 1)) * 0.2
        )
        
        results = {
            'test_size': len(test_questions),
            'retrieval_quality': retrieval_metrics,
            'answer_quality': answer_metrics,
            'response_time': time_metrics,
            'overall_score': overall_score,
            'recommendations': self._generate_recommendations(retrieval_metrics, answer_metrics, time_metrics)
        }
        
        return results
    
    def _generate_recommendations(self, retrieval_metrics: Dict, answer_metrics: Dict, time_metrics: Dict) -> List[str]:
        """Генерирует рекомендации по улучшению системы"""
        recommendations = []
        
        if retrieval_metrics['f1_score'] < 0.7:
            recommendations.append("Низкое качество поиска. Рекомендуется улучшить эмбеддинги или увеличить размер индекса.")
        
        if answer_metrics['semantic_similarity'] < 0.6:
            recommendations.append("Низкое семантическое сходство ответов. Рекомендуется улучшить модель генерации.")
        
        if time_metrics['avg_response_time'] > 3.0:
            recommendations.append("Медленное время отклика. Рекомендуется оптимизировать поиск или использовать более быстрые модели.")
        
        if answer_metrics['avg_answer_length'] < 20:
            recommendations.append("Короткие ответы. Рекомендуется улучшить промптинг или модель генерации.")
        
        if not recommendations:
            recommendations.append("Система показывает хорошие результаты. Рекомендуется регулярное тестирование.")
        
        return recommendations
    
    def save_evaluation_report(self, results: Dict[str, Any], output_path: str = "evaluation_report.json"):
        """Сохраняет отчет об оценке"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Отчет об оценке сохранен: {output_path}")
