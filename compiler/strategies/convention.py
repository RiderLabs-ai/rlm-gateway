"""Sample similar patterns for new features."""

import logging

from compiler.pack import ContextPack, ConventionEntry
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.strategy.convention")


def run(signals: TaskSignals, index, pack: ContextPack, config: dict):
    """Semantic search with k=3, label results as existing pattern examples."""
    threshold = config.get("compiler", {}).get("semantic_threshold", 0.40)

    results = index.embeddings.search(
        query=signals.scope_hint,
        k=3,
        threshold=threshold,
    )

    for chunk_text, file_path, score in results:
        pack.add_convention(ConventionEntry(
            file_path=file_path,
            content=chunk_text,
        ))
