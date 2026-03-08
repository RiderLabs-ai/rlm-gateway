# RLM Gateway

A local Python service that sits between [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (via [CCR](https://github.com/musistudio/claude-code-router)) and the [Moonshot API](https://platform.moonshot.ai) (Kimi K2.5). It intercepts every request, analyzes your codebase, builds a compact context pack of the most relevant code, and prepends it to the prompt before forwarding to the model.

The result: the downstream model sees exactly the code it needs — not your entire codebase — so it responds as if it's been on your project for years.

```
Claude Code → CCR → RLM Gateway → Moonshot API → Kimi K2.5
                        ↑
                   repo index
                   context pack

Ollama runs locally for embeddings only (nomic-embed-text)
```

## How It Works

On every incoming request, the gateway:

1. **Extracts signals** from the prompt — task type (refactor, debug, add feature, etc.), symbol names, file paths mentioned
2. **Compiles a context pack** by running up to 6 strategies against the repo index:
   - **Symbol lookup** — resolves function/class definitions and call sites via tree-sitter AST
   - **File expansion** — includes mentioned files and their direct imports
   - **Semantic search** — finds related code chunks via embeddings (Ollama + sqlite-vec)
   - **Blast radius** — maps upstream dependents for refactors
   - **Convention sampling** — finds similar patterns for new feature implementation
   - **Git context** — recent commits on relevant files
3. **Serializes** the pack to XML and prepends it to the system message
4. **Forwards** the enriched request to the Moonshot API, streaming the response back

If anything fails during compilation, the original request is forwarded unmodified — the gateway never blocks your workflow.

## Prerequisites

- **Python 3.11+**
- **CCR**: `npm install -g @musistudio/claude-code-router`
- **Ollama** (for embeddings only — not inference):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull nomic-embed-text
  ```
- **Moonshot API key** from [platform.moonshot.ai](https://platform.moonshot.ai)

## Installation

```bash
cd rlm-gateway
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | HTTP server with OpenAI-compatible endpoints |
| `httpx` | Async streaming proxy to Moonshot API |
| `tree-sitter` + language packages | AST parsing for Python, TypeScript, JavaScript, Go |
| `networkx` | Module dependency graph |
| `ollama` | Embedding generation via local Ollama instance |
| `sqlite-vec` | Vector similarity search for semantic matching |
| `tiktoken` | Token counting and budget management |
| `gitpython` | Per-file git commit history |
| `watchfiles` | Incremental re-indexing on file changes |
| `cachetools` | TTL cache for compiled context packs |
| `pyyaml` | Configuration loading |
| `rich` | Terminal UI — startup banner, progress bars, request logging |

## Configuration

Edit `config.yaml`:

```yaml
repo_path: "/home/you/projects/your-repo"   # absolute path to the repo to index

indexer:
  languages: [typescript, python, javascript, go]
  exclude: [node_modules, .git, dist, __pycache__, "*.min.js", .next, build]
  embedding_model: "nomic-embed-text"
  embedding_base_url: "http://localhost:11434"
  embeddings_db_path: null                    # default: ~/.rlm-gateway/embeddings.db
  watch: true                                 # auto-reindex on file changes

compiler:
  max_pack_tokens: 6000        # max tokens in a context pack
  symbol_max_body_lines: 60    # truncate function bodies beyond this
  max_call_sites: 5            # max call sites per symbol
  semantic_threshold: 0.72     # minimum similarity score for semantic results

downstream:
  base_url: "https://api.moonshot.cn/v1"
  api_key: "your-moonshot-api-key"
  model: "kimi-k2.5"
  timeout_ms: 120000

cache:
  enabled: true
  max_size: 256         # max cached context packs
  ttl_seconds: 300      # cache lifetime

server:
  host: "127.0.0.1"
  port: 9787
```

### Available Moonshot Models

| Model | Notes |
|---|---|
| `kimi-k2.5` | Latest, best coding + agentic performance (recommended) |
| `kimi-k2` | Previous generation, still strong on code |
| `moonshot-v1-8k` | Lighter, faster, smaller context |
| `moonshot-v1-32k` | Mid-range context |
| `moonshot-v1-128k` | Long context for exploration tasks |

Change `downstream.model` in config.yaml and restart to switch.

## Usage

### Start the gateway

```bash
python3 main.py
```

The gateway starts with a rich terminal UI showing a startup banner, a live progress bar during indexing, and a summary panel when ready:

```
  RLM Gateway
  Downstream:  kimi-k2.5 via https://api.moonshot.cn/v1
  Port:        9787
  Indexing:    /home/you/projects/your-repo

  ⠋ Embedding chunks ━━━━━━━━━━━━━━━━━━  120/247 files 0:00:32

  ╭──────── RLM Gateway Ready ────────╮
  │  Files indexed:     1,847         │
  │  Symbols found:     24,310        │
  │  Chunks embedded:   18,440        │
  │  Time taken:        38.2s         │
  │                                   │
  │  Listening on:      http://…:9787 │
  │  Downstream:        Moonshot …    │
  ╰───────────────────────────────────╯
  File watcher active
```

Subsequent starts load cached embeddings from disk (`~/.rlm-gateway/embeddings.db`) and skip the embedding step, reducing startup to a few seconds.

Per-request logs are printed inline:

```
  14:32:05  POST /v1/chat/completions  refactor  4,812 tokens  200 OK
```

### Configure CCR

Create or edit `~/.ccr/config.json`:

```json
{
  "providers": [
    {
      "name": "rlm",
      "baseUrl": "http://127.0.0.1:9787/v1",
      "apiKey": "rlm-local",
      "models": ["rlm-gateway"]
    }
  ],
  "routing": [
    { "match": "*", "provider": "rlm" }
  ]
}
```

Then start Claude Code as normal — all requests will route through the gateway.

## API Endpoints

### `POST /v1/chat/completions`

Main endpoint. OpenAI-compatible. Extracts signals, compiles a context pack, injects it, and forwards to Moonshot API. Supports both streaming and non-streaming responses.

### `GET /health`

Returns `{"status": "ok"}`. Use for liveness checks.

### `POST /admin/preview`

Run the compiler and return the context pack as JSON **without** forwarding to Moonshot. Essential for debugging what context the model sees.

```bash
curl -X POST http://127.0.0.1:9787/admin/preview \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"explain the OrderService.createOrder function"}]}'
```

Returns:
```json
{
  "signals": {
    "raw_prompt": "explain the OrderService.createOrder function",
    "task_type": "explain",
    "symbols": ["OrderService", "createOrder"],
    "file_mentions": []
  },
  "pack_xml": "<context_pack ...>...</context_pack>",
  "pack_tokens": 2841,
  "sections": { ... }
}
```

### `GET /admin/index/status`

Returns index stats:
```json
{
  "file_count": 1847,
  "last_indexed": "2026-03-08T14:32:00",
  "status": "ready"
}
```

### `POST /admin/index/rebuild`

Triggers a full index rebuild. Useful after large changes (branch switches, rebases).

## Project Structure

```
rlm-gateway/
├── main.py                          # Entry point — rich UI, progress bar, start server
├── config.yaml                      # All configuration
├── requirements.txt                 # Python dependencies
├── gateway/
│   ├── server.py                    # FastAPI app with all endpoints + per-request logging
│   ├── extractor.py                 # Parse TaskSignals from messages (no LLM call)
│   └── forwarder.py                 # Async streaming proxy to Moonshot API
├── indexer/
│   ├── indexer.py                   # Orchestrates index build, file watcher, incremental reindex
│   ├── ast_map.py                   # Tree-sitter symbol/import/call-site index
│   ├── dep_graph.py                 # NetworkX module dependency graph
│   ├── embeddings.py                # Persistent chunk embeddings → sqlite-vec (via Ollama)
│   └── git_meta.py                  # Per-file recent git commits
├── compiler/
│   ├── compiler.py                  # Orchestrates strategies → ContextPack
│   ├── pack.py                      # ContextPack builder + XML serializer
│   ├── budget.py                    # Token counting (tiktoken) + trimming
│   └── strategies/
│       ├── symbol.py                # Resolve symbol definitions + call sites
│       ├── file_expand.py           # Expand seed files + 1-level imports
│       ├── semantic.py              # Semantic search over embeddings
│       ├── blast_radius.py          # Upstream dependents (for refactors)
│       ├── convention.py            # Similar patterns (for new features)
│       └── git_ctx.py               # Recent commits on relevant files
└── repl/
    └── pool.py                      # ProcessPoolExecutor worker pool
```

## Context Pack Format

The compiled context is serialized as XML and prepended to the system message:

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

## Embedding Cache

Embeddings are stored persistently on disk at `~/.rlm-gateway/embeddings.db` (configurable via `indexer.embeddings_db_path`). On startup, if the database already has data, the embedding step is skipped entirely — reducing startup from tens of seconds to a few seconds.

When the file watcher detects changes, only the affected file's chunks are re-embedded (incremental reindex). The `/admin/index/rebuild` endpoint forces a full rebuild including embeddings.

To clear the cache and force a fresh build, delete the database file:

```bash
rm ~/.rlm-gateway/embeddings.db
```

## Error Handling

- **Compiler failure**: logs the error, forwards the original request unmodified
- **Index still building**: forwards unmodified with a warning after 10s
- **Strategy timeout**: skips that strategy, continues with the rest
- **Moonshot API unreachable/error**: returns HTTP 502 with the upstream error body for debugging

## Switching to Local vLLM (Optional)

A `start_vllm.sh` script is included for running a local model instead of the Moonshot API. To switch:

1. Start vLLM: `./start_vllm.sh`
2. Update `config.yaml`:
   ```yaml
   downstream:
     base_url: "http://localhost:8000/v1"
     api_key: "placeholder"
     model: "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"
     timeout_ms: 120000
   ```
3. Restart the gateway

Requires an NVIDIA GPU with sufficient VRAM (tested on RTX 3070 Ti 8GB with AWQ 4-bit quantization, 4096 max context).
