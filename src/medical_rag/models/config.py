"""
Конфигурация для медицинской RAG системы
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Конфигурация моделей"""
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    generation_model: str = "microsoft/DialoGPT-medium"
    device: str = "auto"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class RetrievalConfig:
    """Конфигурация поиска"""
    top_k: int = 3
    similarity_threshold: float = 0.5
    chunk_size: int = 512
    overlap: int = 50
    index_type: str = "faiss"  # faiss, chroma, etc.


@dataclass
class DataConfig:
    """Конфигурация данных"""
    dataset_path: str = "data/rag_clean_dataset_v2_filtered.json"
    corpus_path: str = "data/corpus.jsonl"
    processed_docs_dir: str = "data/processed_docs"
    excluded_range: tuple = (30, 60)
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42
    use_corpus: bool = True


@dataclass
class LoggingConfig:
    """Конфигурация логирования"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """Основная конфигурация системы"""
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Пути
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "models")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "cache")
    embeddings_cache_path: Path = field(init=False)
    vector_index_path: Path = field(init=False)
    
    # Настройки системы
    debug: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    cache_embeddings: bool = True
    
    def __post_init__(self):
        """Создает необходимые директории"""
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.embeddings_cache_path = self.cache_dir / "embeddings.sqlite3"
        self.vector_index_path = self.cache_dir / "vector_index.faiss"
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует конфигурацию в словарь"""
        return {
            "model": {
                "embedding_model": self.model.embedding_model,
                "generation_model": self.model.generation_model,
                "device": self.model.device,
                "max_length": self.model.max_length,
                "temperature": self.model.temperature,
                "top_p": self.model.top_p,
                "do_sample": self.model.do_sample
            },
            "retrieval": {
                "top_k": self.retrieval.top_k,
                "similarity_threshold": self.retrieval.similarity_threshold,
                "chunk_size": self.retrieval.chunk_size,
                "overlap": self.retrieval.overlap,
                "index_type": self.retrieval.index_type
            },
            "data": {
                "dataset_path": self.data.dataset_path,
                "corpus_path": self.data.corpus_path,
                "processed_docs_dir": self.data.processed_docs_dir,
                "excluded_range": self.data.excluded_range,
                "test_size": self.data.test_size,
                "validation_size": self.data.validation_size,
                "random_seed": self.data.random_seed,
                "use_corpus": self.data.use_corpus
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file_path": self.logging.file_path,
                "max_file_size": self.logging.max_file_size,
                "backup_count": self.logging.backup_count
            },
            "paths": {
                "base_dir": str(self.base_dir),
                "data_dir": str(self.data_dir),
                "models_dir": str(self.models_dir),
                "logs_dir": str(self.logs_dir),
                "cache_dir": str(self.cache_dir)
            },
            "system": {
                "debug": self.debug,
                "parallel_processing": self.parallel_processing,
                "max_workers": self.max_workers,
                "cache_embeddings": self.cache_embeddings,
                "embeddings_cache_path": str(self.embeddings_cache_path),
                "vector_index_path": str(self.vector_index_path)
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Создает конфигурацию из словаря"""
        model_config = ModelConfig(**data.get("model", {}))
        retrieval_config = RetrievalConfig(**data.get("retrieval", {}))
        data_config = DataConfig(**data.get("data", {}))
        logging_config = LoggingConfig(**data.get("logging", {}))
        
        config = cls(
            model=model_config,
            retrieval=retrieval_config,
            data=data_config,
            logging=logging_config
        )
        
        # Устанавливаем пути
        paths = data.get("paths", {})
        if "base_dir" in paths:
            config.base_dir = Path(paths["base_dir"])
        if "data_dir" in paths:
            config.data_dir = Path(paths["data_dir"])
        if "models_dir" in paths:
            config.models_dir = Path(paths["models_dir"])
        if "logs_dir" in paths:
            config.logs_dir = Path(paths["logs_dir"])
        system = data.get("system", {})
        if "cache_dir" in paths:
            config.cache_dir = Path(paths["cache_dir"])
            config.cache_dir.mkdir(parents=True, exist_ok=True)
        # Обновляем производные пути после переопределения базовых директорий
        embeddings_cache_path = system.get("embeddings_cache_path")
        vector_index_path = system.get("vector_index_path")
        config.embeddings_cache_path = Path(embeddings_cache_path) if embeddings_cache_path else config.cache_dir / "embeddings.sqlite3"
        config.vector_index_path = Path(vector_index_path) if vector_index_path else config.cache_dir / "vector_index.faiss"
        config.embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
        config.vector_index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Системные настройки
        config.debug = system.get("debug", False)
        config.parallel_processing = system.get("parallel_processing", True)
        config.max_workers = system.get("max_workers", 4)
        config.cache_embeddings = system.get("cache_embeddings", True)
        
        return config
    
    def save(self, path: str):
        """Сохраняет конфигурацию в файл"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Config":
        """Загружает конфигурацию из файла"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
