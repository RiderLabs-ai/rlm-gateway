"""Resolve symbol definitions + call sites."""

import logging

from compiler.pack import ContextPack, SymbolEntry
from gateway.extractor import TaskSignals

logger = logging.getLogger("rlm.strategy.symbol")


def run(signals: TaskSignals, index, pack: ContextPack, config: dict):
    """For each symbol in signals, look up definition and call sites."""
    max_body_lines = config.get("compiler", {}).get("symbol_max_body_lines", 60)
    max_call_sites = config.get("compiler", {}).get("max_call_sites", 5)

    for symbol_name in signals.symbols:
        # Look up exact match first, then partial
        defs = index.ast_map.symbols.get(symbol_name, [])

        # Try partial match if no exact match
        if not defs:
            for key, key_defs in index.ast_map.symbols.items():
                if symbol_name in key or key.endswith(f".{symbol_name}"):
                    defs.extend(key_defs)
                    break

        for sym_def in defs:
            body = sym_def.body
            if body:
                lines = body.split("\n")
                if len(lines) > max_body_lines:
                    body = "\n".join(lines[:max_body_lines]) + "\n  // ... truncated"

            # Get call sites
            call_sites_raw = index.ast_map.call_sites.get(sym_def.name, [])
            call_sites = [
                {"file": cs.file_path, "line": str(cs.line)}
                for cs in call_sites_raw[:max_call_sites]
            ]

            pack.add_symbol(SymbolEntry(
                name=sym_def.name,
                file_path=sym_def.file_path,
                line=sym_def.start_line,
                signature=sym_def.signature,
                body=body,
                call_sites=call_sites,
            ))
