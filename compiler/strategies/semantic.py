"""Semantic search over embeddings."""

import logging

from compiler.pack import ContextPack, SemanticChunk
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.strategy.semantic")


def run(signals: TaskSignals, index, pack: ContextPack, config: dict):
    """Search embedding index with scope_hint, add results to pack."""
    threshold = config.get("compiler", {}).get("semantic_threshold", 0.40)

    results = index.embeddings.search(
        query=signals.scope_hint,
        k=8,
        threshold=threshold,
    )

    for chunk_text, file_path, score in results:
        pack.add_semantic_chunk(SemanticChunk(
            text=chunk_text,
            file_path=file_path,
            score=score,
        ))
