"""Parse TaskSignals from incoming message arrays using heuristics."""

import re
from dataclasses import dataclass, field


@dataclass
class TaskSignals:
    raw_prompt: str
    task_type: str = "other"  # refactor | debug | add_feature | explain | test | other
    symbols: list[str] = field(default_factory=list)
    file_mentions: list[str] = field(default_factory=list)
    scope_hint: str = ""

    def __post_init__(self):
        if not self.scope_hint:
            self.scope_hint = self.raw_prompt


_TASK_KEYWORDS = {
    "refactor": ["refactor", "rename", "extract", "move", "reorganize"],
    "debug": ["bug", "error", "fix", "broken", "failing", "crash", "exception", "traceback"],
    "add_feature": ["add", "implement", "create", "build", "new feature"],
    "explain": ["explain", "how does", "what does", "walk me through"],
    "test": ["test", "spec", "coverage", "unit test"],
}

# Matches CamelCase or camelCase identifiers (likely class/function names)
_SYMBOL_RE = re.compile(r'\b([A-Z][a-zA-Z0-9]*(?:\.[a-zA-Z][a-zA-Z0-9]*)*)\b')
_CAMEL_RE = re.compile(r'\b([a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)\b')

# Matches file path patterns (contains / or has a common extension)
_FILE_RE = re.compile(
    r'(?:^|\s)([a-zA-Z0-9_./-]+(?:/[a-zA-Z0-9_./-]+)+)'  # paths with /
    r'|(?:^|\s)([a-zA-Z0-9_.-]+\.(?:ts|tsx|js|jsx|py|go|rs|java|rb|css|scss|html|json|yaml|yml|toml|md))\b'
)


def _detect_task_type(text: str) -> str:
    lower = text.lower()
    for task_type, keywords in _TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return task_type
    return "other"


def _extract_symbols(text: str) -> list[str]:
    symbols = set()
    for m in _SYMBOL_RE.finditer(text):
        sym = m.group(1)
        # Filter out common English words that happen to be capitalized
        if len(sym) > 1 and not sym.isupper():
            symbols.add(sym)
    for m in _CAMEL_RE.finditer(text):
        symbols.add(m.group(1))
    return list(symbols)


def _extract_file_mentions(text: str) -> list[str]:
    files = set()
    for m in _FILE_RE.finditer(text):
        path = m.group(1) or m.group(2)
        if path:
            files.add(path.strip())
    return list(files)


def extract(messages: list[dict]) -> TaskSignals:
    """Extract TaskSignals from an OpenAI-format message array."""
    # Combine user messages to form the raw prompt
    user_parts = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        user_parts.append(part["text"])

    raw_prompt = "\n".join(user_parts)
    if not raw_prompt:
        return TaskSignals(raw_prompt="")

    return TaskSignals(
        raw_prompt=raw_prompt,
        task_type=_detect_task_type(raw_prompt),
        symbols=_extract_symbols(raw_prompt),
        file_mentions=_extract_file_mentions(raw_prompt),
        scope_hint=raw_prompt,
    )
