"""
CLI-скрипт для подготовки корпуса RAG из DOCX документов.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_rag.data.chunker import ChunkConfig, build_corpus
from medical_rag.data.ingestion import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("build_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подготовка корпуса для RAG.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data") / "raw_docs",
        help="Каталог с исходными .docx файлами",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data") / "processed_docs",
        help="Каталог для сохранения промежуточных JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "corpus.jsonl",
        help="Файл с итоговым корпусом (JSONL)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=400,
        help="Размер чанка (слов)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        help="Перекрытие чанков (слов)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    output_path: Path = args.output

    raw_dir.mkdir(parents=True, exist_ok=True)

    docx_files = sorted(raw_dir.glob("*.docx"))
    if not docx_files:
        logger.error("В каталоге %s нет .docx файлов", raw_dir)
        return

    logger.info("Обработка %s документов", len(docx_files))
    processed_paths = ingest_documents(docx_files, processed_dir)

    if not processed_paths:
        logger.error("Не удалось обработать ни один документ")
        return

    logger.info("Строим корпус...")
    config = ChunkConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    corpus = build_corpus(processed_dir, output_path, config)

    # Сохраним краткую статистику
    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "num_documents": len(processed_paths),
        "num_chunks": len(corpus),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
    }
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    logger.info("Готово. Корпус: %s, статистика: %s", output_path, stats_path)


if __name__ == "__main__":
    main()


