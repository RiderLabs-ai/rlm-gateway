"""Tree-sitter based symbol/import/call-site index."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter as ts

logger = logging.getLogger("rlm.ast_map")

# Language mapping: file extension → tree-sitter language name
_EXT_TO_LANG = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".py": "python",
    ".go": "go",
}


def _load_language(lang_name: str) -> ts.Language | None:
    """Load a tree-sitter Language from its individual package."""
    try:
        if lang_name == "python":
            import tree_sitter_python
            return ts.Language(tree_sitter_python.language())
        elif lang_name == "javascript":
            import tree_sitter_javascript
            return ts.Language(tree_sitter_javascript.language())
        elif lang_name == "typescript":
            import tree_sitter_typescript
            return ts.Language(tree_sitter_typescript.language_typescript())
        elif lang_name == "tsx":
            import tree_sitter_typescript
            return ts.Language(tree_sitter_typescript.language_tsx())
        elif lang_name == "go":
            import tree_sitter_go
            return ts.Language(tree_sitter_go.language())
    except ImportError:
        logger.warning(f"tree-sitter-{lang_name} not installed")
    except Exception as e:
        logger.warning(f"Failed to load language {lang_name}: {e}")
    return None


@dataclass
class SymbolDefinition:
    name: str
    kind: str  # "function" | "method" | "class"
    file_path: str
    start_line: int
    end_line: int
    body: str = ""
    signature: str = ""


@dataclass
class ImportInfo:
    source: str  # module/file being imported from
    names: list[str] = field(default_factory=list)  # specific imports
    file_path: str = ""
    line: int = 0


@dataclass
class CallSite:
    symbol_name: str
    file_path: str
    line: int


class ASTMap:
    def __init__(self, languages: list[str]):
        self.languages = set(languages)
        self.symbols: dict[str, list[SymbolDefinition]] = {}
        self.imports: dict[str, list[ImportInfo]] = {}  # keyed by file_path
        self.call_sites: dict[str, list[CallSite]] = {}  # keyed by symbol_name
        self._parsers: dict[str, any] = {}

    def _get_parser(self, lang: str) -> ts.Parser | None:
        if lang not in self._parsers:
            language = _load_language(lang)
            if language is None:
                self._parsers[lang] = None
                return None
            self._parsers[lang] = ts.Parser(language)
        return self._parsers[lang]

    def _detect_language(self, file_path: str) -> str | None:
        ext = Path(file_path).suffix
        lang = _EXT_TO_LANG.get(ext)
        if lang and lang in self.languages:
            return lang
        # Handle tsx separately
        if lang == "tsx" and "typescript" in self.languages:
            return "tsx"
        return lang if lang in self.languages else None

    def index_file(self, file_path: str, source: str):
        """Index a single file, extracting symbols, imports, and call sites."""
        lang = self._detect_language(file_path)
        if not lang:
            return

        try:
            parser = self._get_parser(lang)
            if parser is None:
                return
            tree = parser.parse(source.encode())
            root = tree.root_node
        except Exception as e:
            logger.debug(f"Parse error for {file_path}: {e}")
            return

        lines = source.split("\n")

        # Remove old entries for this file
        self._remove_file(file_path)

        # Extract based on language
        if lang in ("python",):
            self._index_python(root, file_path, lines, source)
        elif lang in ("typescript", "tsx", "javascript"):
            self._index_js_ts(root, file_path, lines, source)
        elif lang == "go":
            self._index_go(root, file_path, lines, source)

    def _remove_file(self, file_path: str):
        """Remove all entries for a given file."""
        for name in list(self.symbols.keys()):
            self.symbols[name] = [s for s in self.symbols[name] if s.file_path != file_path]
            if not self.symbols[name]:
                del self.symbols[name]

        self.imports.pop(file_path, None)

        for name in list(self.call_sites.keys()):
            self.call_sites[name] = [c for c in self.call_sites[name] if c.file_path != file_path]
            if not self.call_sites[name]:
                del self.call_sites[name]

    def _add_symbol(self, sym: SymbolDefinition):
        self.symbols.setdefault(sym.name, []).append(sym)

    def _add_call_site(self, site: CallSite):
        self.call_sites.setdefault(site.symbol_name, []).append(site)

    def _node_text(self, node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # --- Python indexing ---

    def _index_python(self, root, file_path: str, lines: list[str], source: str):
        source_bytes = source.encode()
        file_imports = []

        def walk(node, class_name=None):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    fname = self._node_text(name_node, source_bytes)
                    full_name = f"{class_name}.{fname}" if class_name else fname
                    body_text = self._node_text(node, source_bytes)
                    # Extract signature (first line)
                    sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
                    self._add_symbol(SymbolDefinition(
                        name=full_name,
                        kind="method" if class_name else "function",
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        body=body_text,
                        signature=sig_line.strip(),
                    ))

            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    cname = self._node_text(name_node, source_bytes)
                    self._add_symbol(SymbolDefinition(
                        name=cname,
                        kind="class",
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))
                    for child in node.children:
                        walk(child, class_name=cname)
                    return  # Don't recurse children again

            elif node.type in ("import_statement", "import_from_statement"):
                imp = self._parse_python_import(node, file_path, source_bytes)
                if imp:
                    file_imports.append(imp)

            elif node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node:
                    call_name = self._node_text(func_node, source_bytes)
                    self._add_call_site(CallSite(
                        symbol_name=call_name,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                    ))

            for child in node.children:
                walk(child, class_name=class_name)

        walk(root)
        if file_imports:
            self.imports[file_path] = file_imports

    def _parse_python_import(self, node, file_path: str, source_bytes: bytes) -> ImportInfo | None:
        text = self._node_text(node, source_bytes)
        if text.startswith("from "):
            parts = text.split()
            if len(parts) >= 4:  # from X import Y
                source = parts[1]
                names = [n.strip().rstrip(",") for n in parts[3:]]
                return ImportInfo(source=source, names=names, file_path=file_path, line=node.start_point[0] + 1)
        elif text.startswith("import "):
            parts = text.split()
            if len(parts) >= 2:
                return ImportInfo(source=parts[1].rstrip(","), names=[], file_path=file_path, line=node.start_point[0] + 1)
        return None

    # --- JavaScript/TypeScript indexing ---

    def _index_js_ts(self, root, file_path: str, lines: list[str], source: str):
        source_bytes = source.encode()
        file_imports = []

        def walk(node, class_name=None):
            # Function declarations
            if node.type in ("function_declaration", "method_definition"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    fname = self._node_text(name_node, source_bytes)
                    full_name = f"{class_name}.{fname}" if class_name else fname
                    body_text = self._node_text(node, source_bytes)
                    sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
                    self._add_symbol(SymbolDefinition(
                        name=full_name,
                        kind="method" if class_name else "function",
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        body=body_text,
                        signature=sig_line.strip(),
                    ))

            # Arrow functions assigned to variables
            elif node.type == "lexical_declaration":
                for child in node.children:
                    if child.type == "variable_declarator":
                        name_n = child.child_by_field_name("name")
                        value_n = child.child_by_field_name("value")
                        if name_n and value_n and value_n.type == "arrow_function":
                            fname = self._node_text(name_n, source_bytes)
                            body_text = self._node_text(node, source_bytes)
                            sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
                            self._add_symbol(SymbolDefinition(
                                name=fname,
                                kind="function",
                                file_path=file_path,
                                start_line=node.start_point[0] + 1,
                                end_line=node.end_point[0] + 1,
                                body=body_text,
                                signature=sig_line.strip(),
                            ))

            # Class declarations
            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    cname = self._node_text(name_node, source_bytes)
                    self._add_symbol(SymbolDefinition(
                        name=cname,
                        kind="class",
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    ))
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        for child in body_node.children:
                            walk(child, class_name=cname)
                    return

            # Import statements
            elif node.type == "import_statement":
                imp = self._parse_js_import(node, file_path, source_bytes)
                if imp:
                    file_imports.append(imp)

            # Call expressions
            elif node.type == "call_expression":
                func_node = node.child_by_field_name("function")
                if func_node:
                    call_name = self._node_text(func_node, source_bytes)
                    self._add_call_site(CallSite(
                        symbol_name=call_name,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                    ))

            for child in node.children:
                walk(child, class_name=class_name)

        walk(root)
        if file_imports:
            self.imports[file_path] = file_imports

    def _parse_js_import(self, node, file_path: str, source_bytes: bytes) -> ImportInfo | None:
        source_node = node.child_by_field_name("source")
        if source_node:
            source_text = self._node_text(source_node, source_bytes).strip("'\"")
            names = []
            for child in node.children:
                if child.type == "import_clause":
                    for spec in child.children:
                        if spec.type == "identifier":
                            names.append(self._node_text(spec, source_bytes))
                        elif spec.type == "named_imports":
                            for imp_spec in spec.children:
                                if imp_spec.type == "import_specifier":
                                    name_n = imp_spec.child_by_field_name("name")
                                    if name_n:
                                        names.append(self._node_text(name_n, source_bytes))
            return ImportInfo(source=source_text, names=names, file_path=file_path, line=node.start_point[0] + 1)
        return None

    # --- Go indexing ---

    def _index_go(self, root, file_path: str, lines: list[str], source: str):
        source_bytes = source.encode()
        file_imports = []

        def walk(node):
            if node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    fname = self._node_text(name_node, source_bytes)
                    body_text = self._node_text(node, source_bytes)
                    sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
                    self._add_symbol(SymbolDefinition(
                        name=fname,
                        kind="function",
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        body=body_text,
                        signature=sig_line.strip(),
                    ))

            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    fname = self._node_text(name_node, source_bytes)
                    # Try to get receiver type
                    receiver = node.child_by_field_name("receiver")
                    rcv_name = ""
                    if receiver:
                        for child in receiver.children:
                            if child.type == "parameter_declaration":
                                type_node = child.child_by_field_name("type")
                                if type_node:
                                    rcv_name = self._node_text(type_node, source_bytes).lstrip("*")
                    full_name = f"{rcv_name}.{fname}" if rcv_name else fname
                    body_text = self._node_text(node, source_bytes)
                    sig_line = lines[node.start_point[0]] if node.start_point[0] < len(lines) else ""
                    self._add_symbol(SymbolDefinition(
                        name=full_name,
                        kind="method",
                        file_path=file_path,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                        body=body_text,
                        signature=sig_line.strip(),
                    ))

            elif node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name_n = child.child_by_field_name("name")
                        if name_n:
                            self._add_symbol(SymbolDefinition(
                                name=self._node_text(name_n, source_bytes),
                                kind="class",
                                file_path=file_path,
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                            ))

            elif node.type == "import_declaration":
                for child in node.children:
                    if child.type == "import_spec":
                        path_node = child.child_by_field_name("path")
                        if path_node:
                            imp_path = self._node_text(path_node, source_bytes).strip('"')
                            file_imports.append(ImportInfo(
                                source=imp_path, names=[], file_path=file_path,
                                line=child.start_point[0] + 1,
                            ))
                    elif child.type == "import_spec_list":
                        for spec in child.children:
                            if spec.type == "import_spec":
                                path_node = spec.child_by_field_name("path")
                                if path_node:
                                    imp_path = self._node_text(path_node, source_bytes).strip('"')
                                    file_imports.append(ImportInfo(
                                        source=imp_path, names=[], file_path=file_path,
                                        line=spec.start_point[0] + 1,
                                    ))

            elif node.type == "call_expression":
                func_node = node.child_by_field_name("function")
                if func_node:
                    call_name = self._node_text(func_node, source_bytes)
                    self._add_call_site(CallSite(
                        symbol_name=call_name,
                        file_path=file_path,
                        line=node.start_point[0] + 1,
                    ))

            for child in node.children:
                walk(child)

        walk(root)
        if file_imports:
            self.imports[file_path] = file_imports

    @property
    def symbol_count(self) -> int:
        return sum(len(defs) for defs in self.symbols.values())
