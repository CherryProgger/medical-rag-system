import json
from pathlib import Path

import pytest

from medical_rag.data.data_processor import DataProcessor
from medical_rag.models.config import Config


def _make_config(tmp_path: Path) -> Config:
    config = Config()
    config.data.corpus_path = str(tmp_path / "corpus.jsonl")
    config.data.dataset_path = str(tmp_path / "dataset.json")
    config.data.processed_docs_dir = str(tmp_path / "processed")
    config.data.use_corpus = True
    config.data_dir = tmp_path
    config.cache_dir = tmp_path / "cache"
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    config.embeddings_cache_path = config.cache_dir / "embeddings.sqlite3"
    config.vector_index_path = config.cache_dir / "vector_index.faiss"
    return config


def _write_corpus(entries, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for entry in entries:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")


def test_load_corpus_entries(tmp_path: Path):
    config = _make_config(tmp_path)
    corpus_entries = [
        {
            "id": "doc1::sec00::chunk000",
            "text": "Первый тестовый чанк.",
            "metadata": {
                "source_file": "doc1.docx",
                "section_title": "Введение",
                "chunk_index": 0,
                "hash": "hash-1",
            },
        },
        {
            "id": "doc1::sec00::chunk001",
            "text": "Второй тестовый чанк.",
            "metadata": {
                "source_file": "doc1.docx",
                "section_title": "Описание",
                "chunk_index": 1,
                "hash": "hash-2",
            },
        },
    ]
    _write_corpus(corpus_entries, Path(config.data.corpus_path))

    processor = DataProcessor(config)
    loaded = processor.load_corpus_entries()

    assert len(loaded) == 2
    assert processor.corpus_entries == loaded
    assert loaded[0]["metadata"]["section_title"] == "Введение"


def test_create_documents_from_corpus(tmp_path: Path):
    config = _make_config(tmp_path)
    entries = [
        {
            "id": "doc2::sec01::chunk000",
            "text": "Содержимое чанка для проверки документов.",
            "metadata": {
                "source_file": "doc2.docx",
                "section_title": "Симптомы",
                "chunk_index": 3,
                "hash": "hash-abc",
                "category": "symptoms",
            },
        }
    ]

    processor = DataProcessor(config)
    documents = processor.create_documents_from_corpus(entries)

    assert len(documents) == 1
    document = documents[0]
    assert document.id == "doc2::sec01::chunk000"
    assert document.content.startswith("Содержимое чанка")
    assert document.metadata.source_file == "doc2.docx"
    assert document.metadata.section == "Симптомы"
    assert document.metadata.extra["chunk_index"] == 3
    assert document.get_cache_key() == "hash-abc"


def test_get_documents_fallback_to_processed_dataset(tmp_path: Path):
    config = _make_config(tmp_path)
    # не создаём файл корпуса, чтобы сработал fallback
    dataset_payload = {
        "dataset_info": {"total_pairs": 1},
        "pairs": [
            {
                "id": "pair-1",
                "question": "Что такое тестовый вопрос?",
                "answer": "Это тестовый ответ.",
                "metadata": {"category": "general"},
                "source": {"file": "doc3.docx", "section": "Общее", "confidence": "medium"},
            }
        ],
        "validation_notes": [],
    }
    with Path(config.data.dataset_path).open("w", encoding="utf-8") as file:
        json.dump(dataset_payload, file, ensure_ascii=False)

    processor = DataProcessor(config)
    documents = processor.get_documents()

    assert len(documents) == 1
    doc = documents[0]
    assert doc.id == "pair-1"
    assert doc.question == "Что такое тестовый вопрос?"
    assert doc.metadata.source_file == "doc3.docx"
    assert doc.metadata.extra["pair_id"] == "pair-1"

