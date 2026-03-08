# RLM Gateway — Local Build Spec

## What This Is

A local Python service that sits between Claude Code and a local model (via vLLM). It intercepts every request, runs a program over your repo to build a compact context pack, prepends it to the prompt, and forwards to the model. The model sees only what's relevant — not your entire codebase.

```
Claude Code → CCR → RLM Gateway (this) → vLLM → local model
                         ↑
                    repo index
                    REPL workspace
                    context pack
```

---

## Prerequisites (install separately before running)

- **CCR**: `npm install -g @musistudio/claude-code-router`
- **vLLM**: running locally, serving any OpenAI-compatible model
- **Ollama**: running locally with `nomic-embed-text` pulled (for embeddings)
- **Python 3.11+**

---

## Project Layout

```
rlm-gateway/
├── main.py
├── config.yaml
├── requirements.txt
├── gateway/
│   ├── server.py        # FastAPI app, OpenAI-compatible endpoint
│   ├── extractor.py     # Parse TaskSignals from incoming messages
│   └── forwarder.py     # Async streaming proxy to vLLM
├── indexer/
│   ├── indexer.py       # Boot-time index build + file watcher
│   ├── ast_map.py       # Tree-sitter symbol/import index
│   ├── dep_graph.py     # Module dependency graph (NetworkX)
│   ├── embeddings.py    # Chunk embeddings → sqlite-vec
│   └── git_meta.py      # Per-file recent commits
├── compiler/
│   ├── compiler.py      # Orchestrates strategies → ContextPack
│   ├── pack.py          # ContextPack builder + XML serializer
│   ├── budget.py        # Token counting + trimming
│   └── strategies/
│       ├── symbol.py        # Resolve symbol definitions + call sites
│       ├── file_expand.py   # Expand seed files + 1-level imports
│       ├── semantic.py      # Semantic search over embeddings
│       ├── blast_radius.py  # What imports the target (for refactors)
│       ├── convention.py    # Sample similar patterns (for new features)
│       └── git_ctx.py       # Recent commits on relevant files
└── repl/
    └── pool.py          # ProcessPoolExecutor worker pool
```

---

## config.yaml

```yaml
repo_path: "/absolute/path/to/your/repo"

indexer:
  languages: [typescript, python, javascript, go]
  exclude: [node_modules, .git, dist, __pycache__, "*.min.js", .next, build]
  embedding_model: "nomic-embed-text"   # via Ollama
  watch: true

compiler:
  max_pack_tokens: 6000
  symbol_max_body_lines: 60
  max_call_sites: 5
  semantic_threshold: 0.72

repl_pool:
  size: 4
  timeout_ms: 2000

downstream:
  base_url: "http://localhost:8000/v1"   # your vLLM instance
  api_key: "placeholder"
  model: "your-model-name"

cache:
  enabled: true
  max_size: 256
  ttl_seconds: 300

server:
  host: "127.0.0.1"
  port: 8787
```

---

## requirements.txt

```
fastapi
uvicorn[standard]
httpx
pyyaml
tree-sitter
tree-sitter-languages
networkx
sqlite-vec
tiktoken
gitpython
watchfiles
ollama
cachetools
```

---

## Implementation Details

### main.py

Load config.yaml, build the repo index, start the file watcher in a background thread, then start FastAPI with uvicorn.

Log progress during index build. Index build on a ~50k line repo should complete in under 60 seconds.

### gateway/server.py

FastAPI app with these endpoints:

- `POST /v1/chat/completions` — main endpoint (OpenAI-compatible request/response)
- `GET /health` — liveness check
- `POST /admin/preview` — run compiler and return the context pack as JSON without forwarding to vLLM (essential for debugging)
- `GET /admin/index/status` — file count, last indexed timestamp
- `POST /admin/index/rebuild` — trigger full index rebuild

On each `POST /v1/chat/completions`:
1. Call `extractor.extract(messages)` → `TaskSignals`
2. Call `compiler.compile(signals)` → `ContextPack`
3. Serialize pack to XML and prepend as a system message (or prepend to the existing system message if one exists)
4. Forward enriched messages to vLLM via `forwarder.forward()`, streaming the response directly back

### gateway/extractor.py

Parse `TaskSignals` from the message array using heuristics — no LLM call.

```python
@dataclass
class TaskSignals:
    raw_prompt: str
    task_type: str            # "refactor" | "debug" | "add_feature" | "explain" | "test" | "other"
    symbols: list[str]        # CamelCase/camelCase tokens likely to be class or function names
    file_mentions: list[str]  # tokens matching a file path pattern (contains "/" or ".")
    scope_hint: str           # full raw prompt, used as semantic search query
```

Detect `task_type` via keyword matching:
- `refactor`: refactor, rename, extract, move, reorganize
- `debug`: bug, error, fix, broken, failing, crash, exception, traceback
- `add_feature`: add, implement, create, build, new feature
- `explain`: explain, how does, what does, walk me through
- `test`: test, spec, coverage, unit test

### gateway/forwarder.py

Async httpx client. Pass the enriched request to the downstream vLLM endpoint. Handle both streaming (SSE) and non-streaming responses. Replace the Authorization header with the downstream API key from config.

### indexer/indexer.py

On boot, build a full `RepoIndex` by walking `repo_path`, respecting `exclude` patterns. Coordinate `ast_map`, `dep_graph`, `embeddings`, and `git_meta`. Expose a single `RepoIndex` object shared across compiler calls.

Use `watchfiles` to watch `repo_path`. On file change, reindex only the changed files (incremental update).

### indexer/ast_map.py

Use `tree-sitter-languages` (pre-built bindings — do not compile grammars manually).

Per file, index:
- All function/method definitions: name, file path, start line, end line, full body text
- All class definitions: name, file path, line
- All import statements: what is imported, from where
- All call sites: which symbol is called, in which file and line

Primary data structure: `dict[symbol_name, list[SymbolDefinition]]`

### indexer/dep_graph.py

NetworkX `DiGraph`. Edge A → B means "A imports B". Build from the import data in `ast_map`.

Expose:
- `dependents(path)` — files that import this file (1–2 hops upstream)
- `dependencies(path)` — files this file imports directly

### indexer/embeddings.py

Chunk at function-level boundaries (use ast_map function definitions). Embed each chunk via Ollama using `nomic-embed-text`. Store vectors in `sqlite-vec`.

Expose: `search(query: str, k: int, threshold: float) → list[(chunk_text, file_path, score)]`

### indexer/git_meta.py

Use `gitpython`. Per file, retrieve the last N commits that touched it (default N=5): hash, message, author, date.

Expose: `recent_commits(path: str, n: int) → list[CommitSummary]`

### compiler/compiler.py

Orchestrate strategies based on `TaskSignals`, assemble a `ContextPack`.

```python
def compile(signals: TaskSignals, index: RepoIndex) -> ContextPack:
    pack = ContextPack()

    if signals.symbols:
        symbol_strategy.run(signals, index, pack)

    if signals.file_mentions:
        file_expand_strategy.run(signals, index, pack)

    semantic_strategy.run(signals, index, pack)

    if signals.task_type == "refactor":
        blast_radius_strategy.run(signals, index, pack)

    if signals.task_type == "add_feature":
        convention_strategy.run(signals, index, pack)

    if signals.task_type in ("debug", "refactor"):
        git_ctx_strategy.run(signals, index, pack)

    pack.trim(max_tokens=config.compiler.max_pack_tokens)
    return pack
```

### compiler/strategies/symbol.py

For each symbol in `signals.symbols`: look up in ast_map, get the definition (file, line range, body text trimmed to `symbol_max_body_lines`), get up to `max_call_sites` call sites. Add to pack.

### compiler/strategies/file_expand.py

For each file path in `signals.file_mentions`: add its full contents to the pack. Resolve its direct imports (depth=1) and add those too. Skip files already in the pack.

### compiler/strategies/semantic.py

Search the embedding index with `signals.scope_hint`, k=8, using the configured threshold. Add resulting chunks to the pack (deduped by file + line range).

### compiler/strategies/blast_radius.py

For each file already in the pack, get its upstream dependents from `dep_graph` (max 2 hops). Add a lightweight summary for each dependent file (just path + first 3 lines). Don't add full contents.

### compiler/strategies/convention.py

Semantic search with k=3. Add results to the pack labelled as "existing pattern" examples so the model understands local conventions.

### compiler/strategies/git_ctx.py

For each file already in the pack, call `git_meta.recent_commits(path, n=3)`. Add to pack as git context.

### compiler/pack.py

`ContextPack` serializes to XML, prepended to the system prompt:

```xml
<context_pack task_id="..." tokens="4812">
  <symbols>
    <symbol name="OrderService.createOrder" file="src/services/order.ts" line="47">
      <signature>async createOrder(input: CreateOrderInput): Promise&lt;Order&gt;</signature>
      <body><![CDATA[... function body ...]]></body>
      <call_sites>
        <site file="src/api/orders.ts" line="82"/>
      </call_sites>
    </symbol>
  </symbols>
  <files>
    <file path="src/api/orders.ts"><![CDATA[... file contents ...]]></file>
  </files>
  <conventions>
    <example file="src/services/product.ts"><![CDATA[... similar pattern ...]]></example>
  </conventions>
  <git_context>
    <file path="src/services/order.ts">
      <commit hash="a3f82c1" message="Fix race condition" author="rider" date="2 days ago"/>
    </file>
  </git_context>
</context_pack>
```

### compiler/budget.py

Use `tiktoken` (`cl100k_base`) for token counting. `trim(pack, max_tokens)` drops sections by priority until under budget.

Drop order (drop last priority first):
1. git_context
2. conventions
3. semantic chunks
4. files (trim longest first)
5. symbols (trim call sites first, then body lines)

### repl/pool.py

`ProcessPoolExecutor` with `pool_size` workers. For v1, running compiler strategies sequentially in-process is fine — the pool is wired up but strategies run inline. This keeps the initial build simple.

---

## CCR Configuration

Once the gateway is running, configure CCR to point at it (`~/.claude-code-router/config.json` or wherever CCR reads config):

```json
{
  "providers": [
    {
      "name": "rlm",
      "baseUrl": "http://127.0.0.1:8787/v1",
      "apiKey": "rlm-local",
      "models": ["rlm-gateway"]
    }
  ],
  "routing": [
    { "match": "*", "provider": "rlm" }
  ]
}
```

---

## Startup

```bash
python main.py
```

Expected output:
```
[rlm] Loading config from config.yaml
[rlm] Building repo index for /path/to/repo...
[rlm] Indexed 1,847 files, 24,310 symbols, 18,440 chunks (38.2s)
[rlm] File watcher active
[rlm] Gateway ready → http://127.0.0.1:8787
```

---

## Error Handling Rules

- If the compiler throws for any reason: log the error, fall back to forwarding the original request unmodified. Never fail the request to Claude Code.
- If the index is still building at startup: queue incoming requests. After 10s, forward unmodified with a log warning.
- If a strategy times out: skip it, continue with the rest.
- If vLLM is unreachable: return HTTP 502 with a readable JSON error body.

---

## Recommended Build Order

1. **Phase 1 — Passthrough**: `server.py` + `forwarder.py` only. No compilation. Validate the full CCR → gateway → vLLM → streaming round-trip works before touching the indexer.

2. **Phase 2 — Index**: `ast_map.py` + `dep_graph.py` + `indexer.py`. Build and log the index. No injection yet.

3. **Phase 3 — Embeddings + Git**: `embeddings.py` + `git_meta.py`. Complete the index.

4. **Phase 4 — Compiler**: All strategies + `pack.py` + `budget.py` + `compiler.py`. Wire into the request path. Use `/admin/preview` to inspect packs before trusting them.

5. **Phase 5 — Hardening**: File watcher, LRU cache, error fallback, graceful shutdown.
