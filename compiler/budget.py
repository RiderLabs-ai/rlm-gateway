"""Token counting and trimming for ContextPack."""

import logging

import tiktoken

logger = logging.getLogger("rlm.budget")

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def trim(pack, max_tokens: int):
    """
    Drop sections by priority until pack is under budget.

    Drop order (drop last priority first):
    1. git_context
    2. conventions
    3. semantic chunks
    4. files (trim longest first)
    5. symbols (trim call sites first, then body lines)
    """
    # Count current tokens
    xml = pack.to_xml()
    current = count_tokens(xml)
    pack.token_count = current

    if current <= max_tokens:
        return

    # 1. Drop git context
    if pack.git_context and current > max_tokens:
        while pack.git_context and current > max_tokens:
            pack.git_context.pop()
            xml = pack.to_xml()
            current = count_tokens(xml)
            pack.token_count = current

    # 2. Drop conventions
    if pack.conventions and current > max_tokens:
        while pack.conventions and current > max_tokens:
            pack.conventions.pop()
            xml = pack.to_xml()
            current = count_tokens(xml)
            pack.token_count = current

    # 3. Drop semantic chunks
    if pack.semantic_chunks and current > max_tokens:
        # Drop lowest-scored first
        pack.semantic_chunks.sort(key=lambda c: c.score, reverse=True)
        while pack.semantic_chunks and current > max_tokens:
            pack.semantic_chunks.pop()
            xml = pack.to_xml()
            current = count_tokens(xml)
            pack.token_count = current

    # 4. Trim files (longest first)
    if pack.files and current > max_tokens:
        pack.files.sort(key=lambda f: len(f.content), reverse=True)
        while pack.files and current > max_tokens:
            pack.files.pop(0)  # Remove longest
            xml = pack.to_xml()
            current = count_tokens(xml)
            pack.token_count = current

    # 5. Trim symbols (call sites first, then body)
    if pack.symbols and current > max_tokens:
        # First strip call sites
        for sym in pack.symbols:
            if current <= max_tokens:
                break
            sym.call_sites = []
            xml = pack.to_xml()
            current = count_tokens(xml)
            pack.token_count = current

        # Then trim body lines
        for sym in pack.symbols:
            if current <= max_tokens:
                break
            if sym.body:
                lines = sym.body.split("\n")
                # Keep reducing body until under budget
                while len(lines) > 5 and current > max_tokens:
                    lines = lines[:len(lines) // 2]
                    sym.body = "\n".join(lines) + "\n  // ... trimmed"
                    xml = pack.to_xml()
                    current = count_tokens(xml)
                    pack.token_count = current

        # Last resort: drop symbols entirely
        while pack.symbols and current > max_tokens:
            pack.symbols.pop()
            xml = pack.to_xml()
            current = count_tokens(xml)
            pack.token_count = current

    pack.token_count = current
