"""Chunk embeddings via Ollama → sqlite-vec storage."""

import json
import logging
import sqlite3
import struct
import tempfile
from pathlib import Path

import ollama
import sqlite_vec

from indexer.ast_map import ASTMap

logger = logging.getLogger("rlm.embeddings")


def _serialize_float32(vec: list[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class EmbeddingStore:
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self._dim = 768  # nomic-embed-text dimension
        self._db_path = Path(tempfile.mkdtemp()) / "embeddings.db"
        self._conn: sqlite3.Connection | None = None
        self._chunks: list[dict] = []  # metadata for each chunk

    def _init_db(self):
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
            USING vec0(embedding float[{self._dim}])
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_meta (
                id INTEGER PRIMARY KEY,
                file_path TEXT,
                start_line INTEGER,
                end_line INTEGER,
                text TEXT
            )
        """)
        self._conn.commit()

    def build(self, ast_map: ASTMap, file_contents: dict[str, str]):
        """Build embeddings from function-level chunks."""
        self._init_db()

        # Clear existing data
        self._conn.execute("DELETE FROM chunk_meta")
        self._conn.execute("DELETE FROM vec_chunks")
        self._conn.commit()
        self._chunks = []

        # Collect chunks from function definitions
        chunks = []
        for symbol_name, defs in ast_map.symbols.items():
            for sym_def in defs:
                if sym_def.kind in ("function", "method") and sym_def.body:
                    chunks.append({
                        "text": sym_def.body[:2000],  # cap chunk size
                        "file_path": sym_def.file_path,
                        "start_line": sym_def.start_line,
                        "end_line": sym_def.end_line,
                    })

        if not chunks:
            logger.info("No chunks to embed")
            return

        # Embed in batches
        batch_size = 32
        chunk_id = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]

            try:
                response = ollama.embed(model=self.model, input=texts)
                embeddings = response.get("embeddings", [])
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                continue

            for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
                self._conn.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (chunk_id, _serialize_float32(emb)),
                )
                self._conn.execute(
                    "INSERT INTO chunk_meta(id, file_path, start_line, end_line, text) VALUES (?, ?, ?, ?, ?)",
                    (chunk_id, chunk["file_path"], chunk["start_line"], chunk["end_line"], chunk["text"]),
                )
                chunk_id += 1

        self._conn.commit()
        self._chunks = chunks
        logger.info(f"Embedded {chunk_id} chunks")

    def search(self, query: str, k: int = 8, threshold: float = 0.72) -> list[tuple[str, str, float]]:
        """Search for similar chunks. Returns (chunk_text, file_path, score)."""
        if not self._conn:
            return []

        try:
            response = ollama.embed(model=self.model, input=[query])
            query_emb = response.get("embeddings", [[]])[0]
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []

        if not query_emb:
            return []

        query_bytes = _serialize_float32(query_emb)

        rows = self._conn.execute(
            """
            SELECT v.rowid, v.distance, m.text, m.file_path, m.start_line, m.end_line
            FROM vec_chunks v
            JOIN chunk_meta m ON v.rowid = m.id
            WHERE v.embedding MATCH ?
            ORDER BY v.distance
            LIMIT ?
            """,
            (query_bytes, k),
        ).fetchall()

        results = []
        for row in rows:
            rowid, distance, text, file_path, start_line, end_line = row
            # sqlite-vec returns L2 distance; convert to similarity-ish score
            # Lower distance = more similar. Use 1/(1+d) as score.
            score = 1.0 / (1.0 + distance)
            if score >= threshold:
                results.append((text, file_path, score))

        return results

    @property
    def chunk_count(self) -> int:
        if not self._conn:
            return 0
        try:
            row = self._conn.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()
            return row[0] if row else 0
        except Exception:
            return 0

    def close(self):
        if self._conn:
            self._conn.close()
