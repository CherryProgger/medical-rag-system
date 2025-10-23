"""
Главный файл для запуска медицинской RAG системы
Профессиональная версия с ООП архитектурой
"""

import argparse
import sys
import logging
from pathlib import Path
import json

# Добавляем путь к модулям
sys.path.append(str(Path(__file__).parent / "src"))

from medical_rag.core.rag_system import MedicalRAGSystem
from medical_rag.models.config import Config
from medical_rag.data.data_processor import DataProcessor


def setup_logging(level: str = "INFO"):
    """Настраивает логирование"""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/rag_system.log")
        ]
    )


def check_dependencies():
    """Проверяет наличие необходимых зависимостей"""
    required_packages = {
        "torch": "torch",
        "transformers": "transformers", 
        "sentence_transformers": "sentence_transformers",
        "faiss": "faiss",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "sklearn",
        "streamlit": "streamlit"
    }
    
    missing_packages = []
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Отсутствуют пакеты: {missing_packages}")
        print("Установите зависимости: pip install -r requirements.txt")
        return False
    
    print("✅ Все зависимости установлены")
    return True


def check_data_files():
    """Проверяет наличие необходимых файлов данных"""
    required_files = [
        "rag_clean_dataset_v2_filtered.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        print("Запустите предобработку данных: python data_preprocessing.py")
        return False
    
    print("✅ Все файлы данных найдены")
    return True


def load_config(config_path: str = None) -> Config:
    """Загружает конфигурацию"""
    if config_path and Path(config_path).exists():
        return Config.load(config_path)
    elif Path("config/default.json").exists():
        return Config.load("config/default.json")
    else:
        return Config()


def run_web_interface(config: Config):
    """Запускает веб-интерфейс"""
    print("Запуск веб-интерфейса...")
    import subprocess
    subprocess.run(["streamlit", "run", "src/medical_rag/interfaces/web_interface.py", "--server.port", "8501"])


def run_cli_demo(config: Config):
    """Запускает демонстрацию в командной строке"""
    print("Инициализация RAG системы...")
    rag = MedicalRAGSystem(config)
    rag.initialize()
    
    print("\n" + "="*60)
    print("🏥 МЕДИЦИНСКАЯ RAG СИСТЕМА - ПРОФЕССИОНАЛЬНАЯ ВЕРСИЯ")
    print("="*60)
    print("Система готова к работе!")
    print("Введите 'quit' для выхода")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Ваш вопрос: ").strip()
            
            if question.lower() in ['quit', 'exit', 'выход']:
                print("До свидания! 👋")
                break
            
            if not question:
                continue
            
            print("🔍 Поиск ответа...")
            response = rag.answer_question(question)
            
            print(f"\n📋 Ответ:")
            print(f"{response.answer}")
            
            print(f"\n📊 Статистика:")
            print(f"  • Время обработки: {response.metadata.processing_time:.2f}с")
            print(f"  • Найдено документов: {response.metadata.num_documents_found}")
            print(f"  • Релевантность: {response.metadata.best_similarity_score:.3f}")
            print(f"  • Уверенность: {response.metadata.confidence_level}")
            
            if response.relevant_documents:
                print(f"\n📚 Источники:")
                for i, doc in enumerate(response.relevant_documents[:2], 1):
                    print(f"  {i}. {doc.question[:50]}... (score: {doc.similarity_score:.3f})")
            
            if response.warnings:
                print(f"\n⚠️ Предупреждения:")
                for warning in response.warnings:
                    print(f"  • {warning}")
            
        except KeyboardInterrupt:
            print("\n\nДо свидания! 👋")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


def run_evaluation(config: Config):
    """Запускает оценку системы"""
    print("Инициализация системы для оценки...")
    rag = MedicalRAGSystem(config)
    rag.initialize()
    
    print("Создание оценщика...")
    from medical_rag.services.evaluation_service import EvaluationService
    evaluator = EvaluationService(rag)
    
    print("Запуск оценки...")
    results = evaluator.run_full_evaluation()
    
    print("\n" + "="*50)
    print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*50)
    
    print(f"Общий балл: {results['overall_score']:.3f}/1.0")
    print(f"Тестовых вопросов: {results['test_size']}")
    
    print(f"\n🔍 Качество поиска:")
    print(f"  Precision: {results['retrieval_quality']['precision']:.3f}")
    print(f"  Recall: {results['retrieval_quality']['recall']:.3f}")
    print(f"  F1-Score: {results['retrieval_quality']['f1_score']:.3f}")
    
    print(f"\n💬 Качество ответов:")
    print(f"  Семантическое сходство: {results['answer_quality']['semantic_similarity']:.3f}")
    print(f"  Пересечение ключевых слов: {results['answer_quality']['keyword_overlap']:.3f}")
    print(f"  Средняя длина ответа: {results['answer_quality']['avg_answer_length']:.1f} слов")
    
    print(f"\n⏱️ Время отклика:")
    print(f"  Среднее время: {results['response_time']['avg_response_time']:.2f}с")
    print(f"  Медианное время: {results['response_time']['median_response_time']:.2f}с")
    
    print(f"\n💡 Рекомендации:")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Сохранение отчета
    evaluator.save_evaluation_report(results, "evaluation_report.json")
    print(f"\n📄 Отчет сохранен: evaluation_report.json")


def run_data_preprocessing():
    """Запускает предобработку данных"""
    print("Предобработка данных...")
    
    config = load_config()
    processor = DataProcessor(config)
    
    # Загружаем исходный датасет
    processor.load_raw_dataset("data/rag_clean_dataset_v2.json")
    
    # Создаем обработанный датасет
    processed_data = processor.create_processed_dataset()
    
    # Сохраняем обработанный датасет
    output_path = processor.save_processed_dataset()
    
    # Показываем статистику
    stats = processor.get_statistics()
    print(f"\n📊 Статистика:")
    print(f"  • Всего пар: {stats['total_pairs']}")
    print(f"  • Исключено: {stats['excluded_count']}")
    print(f"  • Категории: {stats['categories']}")
    print(f"  • Темы: {stats['topics']}")
    print(f"  • Сложность: {stats['difficulties']}")
    
    print(f"\n✅ Обработанный датасет сохранен: {output_path}")


def run_tests():
    """Запускает тесты"""
    print("Запуск тестов...")
    import subprocess
    result = subprocess.run(["python", "-m", "pytest", "tests/", "-v"], capture_output=True, text=True)
    
    print("Результаты тестов:")
    print(result.stdout)
    if result.stderr:
        print("Ошибки:")
        print(result.stderr)
    
    return result.returncode == 0


def show_system_info():
    """Показывает информацию о системе"""
    print("🏥 МЕДИЦИНСКАЯ RAG СИСТЕМА - ПРОФЕССИОНАЛЬНАЯ ВЕРСИЯ")
    print("="*60)
    print("Версия: 2.0.0")
    print("Архитектура: ООП, модульная")
    print("Данные: Медицинская документация по сосудистым заболеваниям")
    print("Технологии: RAG, Sentence Transformers, FAISS, PyTorch")
    print()
    print("📁 Структура проекта:")
    print("  • src/medical_rag/ - Основные модули")
    print("  • tests/ - Тесты (unit, integration, e2e)")
    print("  • config/ - Конфигурационные файлы")
    print("  • docs/ - Документация")
    print("  • examples/ - Примеры использования")
    print()
    print("🚀 Режимы запуска:")
    print("  • python main.py --web - Веб-интерфейс")
    print("  • python main.py --cli - Командная строка")
    print("  • python main.py --eval - Оценка системы")
    print("  • python main.py --preprocess - Предобработка данных")
    print("  • python main.py --test - Запуск тестов")
    print("  • python main.py --info - Информация о системе")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Медицинская RAG система - Профессиональная версия")
    parser.add_argument("--web", action="store_true", help="Запуск веб-интерфейса")
    parser.add_argument("--cli", action="store_true", help="Запуск в режиме командной строки")
    parser.add_argument("--eval", action="store_true", help="Запуск оценки системы")
    parser.add_argument("--preprocess", action="store_true", help="Предобработка данных")
    parser.add_argument("--test", action="store_true", help="Запуск тестов")
    parser.add_argument("--info", action="store_true", help="Показать информацию о системе")
    parser.add_argument("--check", action="store_true", help="Проверить зависимости и файлы")
    parser.add_argument("--config", type=str, help="Путь к файлу конфигурации")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Уровень логирования")
    
    args = parser.parse_args()
    
    # Настройка логирования
    setup_logging(args.log_level)
    
    # Проверка зависимостей и файлов
    if args.check or not any([args.web, args.cli, args.eval, args.preprocess, args.test, args.info]):
        print("🔍 Проверка системы...")
        
        deps_ok = check_dependencies()
        data_ok = check_data_files()
        
        if deps_ok and data_ok:
            print("✅ Система готова к работе!")
        else:
            print("❌ Требуется установка зависимостей или предобработка данных")
            sys.exit(1)
        
        if not any([args.web, args.cli, args.eval, args.preprocess, args.test, args.info]):
            show_system_info()
            return
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Запуск соответствующих режимов
    if args.info:
        show_system_info()
    elif args.preprocess:
        run_data_preprocessing()
    elif args.test:
        success = run_tests()
        sys.exit(0 if success else 1)
    elif args.web:
        run_web_interface(config)
    elif args.cli:
        run_cli_demo(config)
    elif args.eval:
        run_evaluation(config)
    else:
        print("Используйте --help для просмотра доступных опций")


if __name__ == "__main__":
    main()
