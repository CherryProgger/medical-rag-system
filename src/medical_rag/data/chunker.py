"""
Разбиение документов на чанки для RAG.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .ingestion import load_processed_document

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Параметры разбиения документа на чанки."""

    chunk_size: int = 400  # целевое количество слов
    overlap: int = 80
    min_chunk_size: int = 120


def _split_into_words(text: str) -> List[str]:
    return text.split()


def _words_to_text(words: List[str]) -> str:
    return " ".join(words).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_section(
    doc_id: str,
    title: str,
    text: str,
    config: ChunkConfig,
    section_idx: int,
    metadata: Optional[Dict[str, str]] = None,
) -> List[Dict[str, object]]:
    """
    Разбивает конкретный раздел на чанки.
    """
    words = _split_into_words(text)
    if not words:
        return []

    chunks: List[Dict[str, object]] = []
    step = config.chunk_size - config.overlap
    if step <= 0:
        raise ValueError("chunk_size должен быть больше overlap")

    start = 0
    chunk_idx = 0
    while start < len(words):
        end = min(len(words), start + config.chunk_size)
        chunk_words = words[start:end]

        # если получился слишком маленький хвост, приклеиваем его к последнему чанкe
        if len(chunk_words) < config.min_chunk_size and chunks:
            chunks[-1]["text"] = _words_to_text(
                _split_into_words(chunks[-1]["text"]) + chunk_words
            )
            chunks[-1]["hash"] = _hash_text(chunks[-1]["text"])
            break

        chunk_text = _words_to_text(chunk_words)
        chunk_id = f"{doc_id}::sec{section_idx:02d}::chunk{chunk_idx:03d}"
        chunk_metadata = {
            "section_title": title,
            "section_index": section_idx,
            "chunk_index": chunk_idx,
            "num_words": len(chunk_words),
            "hash": _hash_text(chunk_text),
        }
        if metadata:
            chunk_metadata.update(metadata)

        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata,
            }
        )

        chunk_idx += 1
        start += step

    return chunks


def chunk_processed_document(
    processed_path: Path,
    config: ChunkConfig,
) -> List[Dict[str, object]]:
    """
    Разбивает обработанный документ (JSON) на чанки.
    """
    data = load_processed_document(processed_path)
    sections = data.get("sections", [])
    source_file = data.get("source_file")

    all_chunks: List[Dict[str, object]] = []
    for idx, section in enumerate(sections):
        section_text = str(section.get("text", "")).strip()
        if not section_text:
            continue

        section_chunks = chunk_section(
            doc_id=Path(source_file).stem if source_file else processed_path.stem,
            title=str(section.get("title", f"Раздел {idx+1}")),
            text=section_text,
            config=config,
            section_idx=idx,
            metadata={
                "source_file": source_file,
                "section_level": section.get("level"),
                "section_order": section.get("order"),
            },
        )
        all_chunks.extend(section_chunks)

    logger.info(
        "Разделили %s на %s чанков",
        processed_path.name,
        len(all_chunks),
    )
    return all_chunks


def build_corpus(
    processed_dir: Path,
    output_path: Path,
    config: ChunkConfig,
) -> List[Dict[str, object]]:
    """
    Собирает единый корпус на основе обработанных документов.
    """
    corpus: List[Dict[str, object]] = []
    processed_files = sorted(processed_dir.glob("*.json"))
    if not processed_files:
        raise FileNotFoundError(f"В каталоге {processed_dir} нет обработанных документов")

    for processed_file in processed_files:
        corpus.extend(chunk_processed_document(processed_file, config))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with output_path.open("w", encoding="utf-8") as file:
        for entry in corpus:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")

    logger.info("Собран корпус из %s чанков: %s", len(corpus), output_path)
    return corpus


