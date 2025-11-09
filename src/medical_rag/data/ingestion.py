"""
Модуль для извлечения текста из медицинских документов и подготовки корпуса.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from docx2python import docx2python

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSection:
    """Представление выделенного раздела документа."""

    title: str
    text: str
    order: int
    level: int


class DocumentIngestionError(RuntimeError):
    """Ошибка извлечения документа."""


def _run_pandoc(doc_path: Path) -> Optional[str]:
    """
    Конвертирует DOCX в markdown через pandoc.

    Args:
        doc_path: путь к документу

    Returns:
        Markdown строка или None, если pandoc недоступен/произошла ошибка.
    """
    if shutil.which("pandoc") is None:
        logger.debug("Pandoc не найден, используем fallback.")
        return None

    try:
        result = subprocess.run(
            ["pandoc", "--wrap=none", "--from=docx", "--to=markdown", str(doc_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("Pandoc не смог обработать %s: %s", doc_path, exc)
        return None


def _cleanup_markdown(markdown_text: str) -> str:
    """Упрощает markdown и убирает артефакты."""
    text = markdown_text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_docx2python(doc_path: Path) -> str:
    """Fallback извлечение текста без структуры."""
    try:
        parsed = docx2python(doc_path, html=False)
    except Exception as exc:  # pylint: disable=broad-except
        raise DocumentIngestionError(f"docx2python не смог обработать {doc_path}: {exc}") from exc

    # docx2python текст представляет как вложенные списки: документ -> тела -> параграфы -> строки
    paragraphs: List[str] = []
    for body in parsed.body:
        for table in body:
            for row in table:
                for cell in row:
                    joined = " ".join(line.strip() for line in cell if line.strip())
                    if joined:
                        paragraphs.append(joined)
    return "\n\n".join(paragraphs)


def _split_markdown_sections(markdown_text: str) -> List[ExtractedSection]:
    """
    Разбивает markdown на разделы по заголовкам.

    Простой парсер: ищем строки, начинающиеся с '#'.
    """
    sections: List[ExtractedSection] = []
    current_title = "Введение"
    current_level = 1
    buffer: List[str] = []

    lines = markdown_text.splitlines()
    order = 1
    for line in lines:
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if heading_match:
            if buffer:
                sections.append(
                    ExtractedSection(
                        title=current_title.strip(),
                        text="\n".join(buffer).strip(),
                        order=order,
                        level=current_level,
                    )
                )
                order += 1
                buffer = []
            current_level = len(heading_match.group(1))
            current_title = heading_match.group(2)
        else:
            buffer.append(line)

    if buffer:
        sections.append(
            ExtractedSection(
                title=current_title.strip(),
                text="\n".join(buffer).strip(),
                order=order,
                level=current_level,
            )
        )

    # фильтруем пустые разделы
    return [section for section in sections if section.text]


def extract_document(doc_path: Path) -> Dict[str, List[Dict[str, object]]]:
    """
    Извлекает структурированный текст из документа.

    Args:
        doc_path: путь к DOCX

    Returns:
        Словарь с метаданными и разделами.
    """
    if not doc_path.exists():
        raise FileNotFoundError(doc_path)

    markdown = _run_pandoc(doc_path)
    if markdown:
        markdown = _cleanup_markdown(markdown)
        sections = _split_markdown_sections(markdown)
    else:
        logger.info("Используем docx2python для %s", doc_path)
        raw_text = _extract_with_docx2python(doc_path)
        sections = [
            ExtractedSection(title="Документ", text=raw_text, order=1, level=1),
        ]

    serialized_sections = [
        {
            "title": section.title,
            "text": section.text,
            "order": section.order,
            "level": section.level,
        }
        for section in sections
        if section.text.strip()
    ]

    if not serialized_sections:
        raise DocumentIngestionError(f"Не удалось извлечь разделы из {doc_path}")

    return {
        "source_file": doc_path.name,
        "sections": serialized_sections,
    }


def ingest_documents(input_paths: Iterable[Path], output_dir: Path) -> List[Path]:
    """
    Обрабатывает множество документов и сохраняет результат в JSON.

    Args:
        input_paths: список путей к .docx файлам
        output_dir: каталог для сохранения .json

    Returns:
        Список путей к созданным файлам.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[Path] = []

    for doc_path in input_paths:
        try:
            parsed = extract_document(doc_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Не удалось обработать %s: %s", doc_path, exc)
            continue

        output_path = output_dir / f"{doc_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(parsed, file, ensure_ascii=False, indent=2)
        saved_files.append(output_path)
        logger.info("Сохранён обработанный документ: %s", output_path)

    return saved_files


def load_processed_document(processed_path: Path) -> Dict[str, object]:
    """Загружает ранее сохранённый JSON с документом."""
    with processed_path.open("r", encoding="utf-8") as file:
        return json.load(file)


