"""ContextPack builder + XML serializer."""

import uuid
from dataclasses import dataclass, field
from xml.sax.saxutils import escape


@dataclass
class SymbolEntry:
    name: str
    file_path: str
    line: int
    signature: str = ""
    body: str = ""
    call_sites: list[dict] = field(default_factory=list)  # [{file, line}]


@dataclass
class FileEntry:
    path: str
    content: str


@dataclass
class ConventionEntry:
    file_path: str
    content: str


@dataclass
class GitEntry:
    file_path: str
    commits: list[dict] = field(default_factory=list)  # [{hash, message, author, date}]


@dataclass
class SemanticChunk:
    text: str
    file_path: str
    score: float


class ContextPack:
    def __init__(self):
        self.task_id = str(uuid.uuid4())[:8]
        self.symbols: list[SymbolEntry] = []
        self.files: list[FileEntry] = []
        self.conventions: list[ConventionEntry] = []
        self.git_context: list[GitEntry] = []
        self.semantic_chunks: list[SemanticChunk] = []
        self.token_count: int = 0

        # Track what's already in the pack to avoid duplicates
        self._seen_files: set[str] = set()
        self._seen_symbols: set[str] = set()
        self._seen_chunks: set[tuple[str, int]] = set()  # (file, start_line) tuples

    def add_symbol(self, entry: SymbolEntry):
        key = f"{entry.name}@{entry.file_path}"
        if key not in self._seen_symbols:
            self._seen_symbols.add(key)
            self.symbols.append(entry)

    def add_file(self, entry: FileEntry):
        if entry.path not in self._seen_files:
            self._seen_files.add(entry.path)
            self.files.append(entry)

    def add_convention(self, entry: ConventionEntry):
        self.conventions.append(entry)

    def add_git_entry(self, entry: GitEntry):
        self.git_context.append(entry)

    def add_semantic_chunk(self, chunk: SemanticChunk):
        # Dedupe by file+approximate location
        key = (chunk.file_path, hash(chunk.text[:100]))
        if key not in self._seen_chunks:
            self._seen_chunks.add(key)
            self.semantic_chunks.append(chunk)

    def has_file(self, path: str) -> bool:
        return path in self._seen_files

    def all_file_paths(self) -> set[str]:
        """Return all file paths present in the pack (symbols + files)."""
        paths = set(self._seen_files)
        for sym in self.symbols:
            paths.add(sym.file_path)
        return paths

    def to_xml(self) -> str:
        parts = [f'<context_pack task_id="{self.task_id}" tokens="{self.token_count}">']

        # Symbols
        if self.symbols:
            parts.append("  <symbols>")
            for sym in self.symbols:
                parts.append(f'    <symbol name="{escape(sym.name)}" file="{escape(sym.file_path)}" line="{sym.line}">')
                if sym.signature:
                    parts.append(f"      <signature>{escape(sym.signature)}</signature>")
                if sym.body:
                    parts.append(f"      <body><![CDATA[{sym.body}]]></body>")
                if sym.call_sites:
                    parts.append("      <call_sites>")
                    for site in sym.call_sites:
                        parts.append(f'        <site file="{escape(site["file"])}" line="{site["line"]}"/>')
                    parts.append("      </call_sites>")
                parts.append("    </symbol>")
            parts.append("  </symbols>")

        # Files
        if self.files:
            parts.append("  <files>")
            for f in self.files:
                parts.append(f'    <file path="{escape(f.path)}"><![CDATA[{f.content}]]></file>')
            parts.append("  </files>")

        # Semantic chunks
        if self.semantic_chunks:
            parts.append("  <semantic_matches>")
            for chunk in self.semantic_chunks:
                parts.append(f'    <match file="{escape(chunk.file_path)}" score="{chunk.score:.2f}"><![CDATA[{chunk.text}]]></match>')
            parts.append("  </semantic_matches>")

        # Conventions
        if self.conventions:
            parts.append("  <conventions>")
            for conv in self.conventions:
                parts.append(f'    <example file="{escape(conv.file_path)}"><![CDATA[{conv.content}]]></example>')
            parts.append("  </conventions>")

        # Git context
        if self.git_context:
            parts.append("  <git_context>")
            for g in self.git_context:
                parts.append(f'    <file path="{escape(g.file_path)}">')
                for c in g.commits:
                    parts.append(
                        f'      <commit hash="{escape(c["hash"])}" '
                        f'message="{escape(c["message"])}" '
                        f'author="{escape(c["author"])}" '
                        f'date="{escape(c["date"])}"/>'
                    )
                parts.append("    </file>")
            parts.append("  </git_context>")

        parts.append("</context_pack>")
        return "\n".join(parts)

    def summary(self) -> dict:
        return {
            "symbols": len(self.symbols),
            "files": len(self.files),
            "semantic_chunks": len(self.semantic_chunks),
            "conventions": len(self.conventions),
            "git_entries": len(self.git_context),
            "token_count": self.token_count,
        }
