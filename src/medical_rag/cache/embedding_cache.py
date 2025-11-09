"""
Кэширование эмбеддингов документов.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Простой SQLite-кэш для эмбеддингов документов."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                dimension INTEGER NOT NULL
            )
            """
        )
        self._connection.commit()
        logger.debug("Инициализирован кэш эмбеддингов: %s", self.path)

    def get(self, cache_key: str) -> Optional[list[float]]:
        """
        Возвращает эмбеддинг из кэша.
        """
        cursor = self._connection.execute(
            "SELECT embedding FROM embeddings WHERE cache_key = ?", (cache_key,)
        )
        row = cursor.fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                logger.warning("Не удалось декодировать эмбеддинг для ключа %s", cache_key)
        return None

    def set(self, cache_key: str, embedding: Iterable[float]) -> None:
        """
        Сохраняет эмбеддинг в кэш.
        """
        embedding_list = list(embedding)
        payload = json.dumps(embedding_list)
        dimension = len(embedding_list)
        self._connection.execute(
            """
            INSERT OR REPLACE INTO embeddings (cache_key, embedding, dimension)
            VALUES (?, ?, ?)
            """,
            (cache_key, payload, dimension),
        )
        self._connection.commit()

    def close(self) -> None:
        """Закрывает соединение с кэшем."""
        if self._connection:
            self._connection.close()
            self._connection = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # pylint: disable=broad-except
            pass


