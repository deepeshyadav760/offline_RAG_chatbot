# ISR-GPT

ISR-GPT is a Retrieval-Augmented Generation (RAG) system designed for querying imagery intelligence (IMINT) reports with cited source chunks. It combines PDF ingestion, structured chunking, dense embedding search, classification-based query analysis, cross-encoder reranking, and local LLM generation.

## Project Overview

This repository implements a Python FastAPI backend for ingesting PDF reports and answering analyst-style questions over those reports. The system is optimized for IMINT content, including:

- structured observation tables
- freeform report text
- appendix images
- source citation tracking
- thread-scoped vector collections and folders
- active prompt injection for custom workflows

The core behavior is:

1. Parse PDF reports into structured text, tables, and images
2. Chunk documents into retrieval-ready units
3. Embed chunks with a sentence-transformer embedding model
4. Store chunk embeddings and metadata in ChromaDB
5. Analyze incoming queries for locations, dates, and intent
6. Retrieve candidate chunks from the vector store
7. Rerank candidates with a cross-encoder reranker
8. Generate an answer from a local LLM using the selected chunks

## Architecture

### Core components

- `src/isr_gpt/main.py`
  - FastAPI application entrypoint
  - Initializes configuration, SQLite DB, ChromaDB, embedding manager, reranker, and retrieval pipeline
  - Registers API routers

- `src/isr_gpt/config/settings.py`
  - Runtime settings via `pydantic-settings`
  - Model registry loader for `config/models.yaml`
  - Embedding, LLM, and reranker configuration classes

- `src/isr_gpt/db/database.py`
  - SQLite database initialization and migrations
  - Table schema creation and seeding
  - Foreign key enforcement and WAL mode

- `src/isr_gpt/ingestion/pdf_parser.py`
  - PDF parsing using `PyMuPDF`
  - Extracts text blocks, table rows, headings, and images

- `src/isr_gpt/ingestion/chunker.py`
  - Converts parsed PDF content into retrieval chunks
  - Handles structured IMINT observation tables and freeform text
  - Generates unique chunk IDs and optional evaluation alignment

- `src/isr_gpt/embedding/manager.py`
  - Loads sentence-transformer embedding models
  - Runs embedding inference for documents and queries

- `src/isr_gpt/vectorstore/store.py`
  - Wraps ChromaDB persistent store
  - Manages collections, upserts, queries, and statistics

- `src/isr_gpt/retrieval/retriever.py`
  - Query analysis, embedding search, optional filters, and reranking

- `src/isr_gpt/retrieval/reranker.py`
  - Cross-encoder reranking of query-passage pairs
  - Diversity-aware selection across source reports

- `src/isr_gpt/generation/manager.py`
  - Calls `llama-cpp-python` for GGUF LLM inference
  - Builds chat prompts and extracts citation IDs

## Repo Structure

```
config/                # YAML and prompt config files
data/                  # persistent storage for DB and Chroma
env/                   # local Python environment files
models/                # local model folders for embeddings, llms, rerankers
src/isr_gpt/           # package source code
  api/                 # FastAPI routes and Pydantic schemas
  config/              # environment and model configuration
  db/                  # SQLite database management
  embedding/           # embedding manager
  generation/          # LLM prompt and generation manager
  ingestion/           # PDF parser and chunking logic
  retrieval/           # query analysis and reranking
  vectorstore/         # ChromaDB wrapper and chunk schema
README.md              # this documentation file
requirements.txt       # Python dependencies
pyproject.toml         # packaging and metadata
```

## Installation and Setup

### Dependencies

The project dependencies are declared in `requirements.txt` and `pyproject.toml`.

Required packages include:

- `fastapi`
- `uvicorn[standard]`
- `PyMuPDF`
- `markdown`
- `chromadb`
- `sentence-transformers`
- `pydantic`
- `pydantic-settings`
- `structlog`
- `pyyaml`
- `einops`
- `huggingface_hub`
- `bcrypt`
- `PyJWT`

Optional / evaluation:

- `rouge-score`
- `nltk`

LLM inference requires `llama-cpp-python`, which is installed separately with CUDA support if desired.

### Basic startup

1. Install dependencies from `requirements.txt`.
2. Ensure `config/models.yaml` is configured with your local model paths.
3. Run the server with Uvicorn or the provided launcher.

Example:

```powershell
pip install -r requirements.txt
uvicorn src.isr_gpt.main:app --host 127.0.0.1 --port 8000
```

Or with `start.bat` if available.

## Configuration

### Runtime settings

`src/isr_gpt/config/settings.py` defines runtime configuration via environment variables and `.env` file:

- `reports_dir`
- `data_dir`
- `models_dir`
- `config_dir`
- `api_host`
- `api_port`
- `log_level`

These can be overridden in `config/.env`.

### Model registry

`config/models.yaml` defines embedding, LLM, and reranker models.

#### Embeddings

The active embedding model is configured under `embedding.active`.

Available embedding models include:

- `qwen3-embedding-4b` (active default)
- `qwen3-embedding-8b`
- `qwen3-embedding-0.6b`
- `bge-large-en-v1.5`
- `bge-base-en-v1.5`
- `nomic-embed-text-v1.5`
- `embeddinggemma-300m`
- `nv-embed-v2`

Each embedding entry contains:

- `hf_name`: Hugging Face model name
- `path`: local model path
- `dimensions`: embedding vector dimension
- `max_seq_length`: maximum text length
- `normalize`: whether embeddings are normalized
- `query_prefix`: prefix prepended to query text
- `torch_dtype`: optional dtype for model loading

#### LLMs

The active LLM is configured under `llm.active`.

Available LLMs include:

- `gemma-4-e4b-it-q4` (active default)
- `gemma-4-e4b-it`
- `gemma-4-e4b-it-q4-8gb`
- `gemma-4-27b-a4b-it`
- `gemma-4-27b-a4b-it-32gb`
- `phi-3.5-mini-instruct`
- `llama-3.1-8b-instruct`
- `qwen2.5-7b-instruct`

Each LLM entry contains:

- `hf_repo`, `file_name`, `path`
- `context_length`
- `n_gpu_layers`
- `temperature`
- `max_tokens`
- `chat_format`
- performance tuning options like `flash_attn`, `n_batch`, `n_ubatch`, `use_mmap`, `offload_kqv`, `type_k`, `type_v`

#### Reranker

The reranker model is configured under `reranker.active`.

Active default:

- `ms-marco-MiniLM-L-6-v2`

This is a sentence-transformer cross-encoder used to score query-chunk pairs.

#### Retrieval tuning

The `retrieval` section provides:

- `initial_top_k`
- `final_top_k`
- `exhaustive_final_top_k`
- `score_threshold`

#### Generation prompt

The `generation.system_prompt_path` points to `config/system_prompt.txt`, which defines the LLM assistant behavior and citation rules.

## Data Pipeline

### 1. Ingestion

The ingestion pipeline is implemented in `src/isr_gpt/api/routes_ingest.py` and `src/isr_gpt/api/routes_thread_ingest.py`.

- A PDF is parsed by `PDFParser.parse()` from `src/isr_gpt/ingestion/pdf_parser.py`.
- The parsed document is chunked by `Chunker.chunk()` in `src/isr_gpt/ingestion/chunker.py`.
- Text chunks are embedded using `EmbeddingManager.embed()`.
- Embeddings and metadata are written to ChromaDB via `VectorStore.upsert_chunks()`.

Batch ingestion supports directories of PDFs and optionally derives location metadata from folder names.

### 2. PDF parsing

`PDFParser` uses `PyMuPDF` to extract:

- tables via `page.find_tables()`
- raw text blocks using `page.get_text("dict")`
- image bytes and bounding boxes
- candidate observation rows when tabular detection fails

It creates a `ParsedDocument` with `ParsedPage` entries, containing:

- `text_blocks`
- `image_blocks`
- `table_rows`

This parser is built for IMINT reports where structured observation tables and headings are common.

### 3. Chunking

`Chunker` in `src/isr_gpt/ingestion/chunker.py` converts parsed pages into retrieval chunks.

It supports two main chunking modes:

- `structured`: one chunk per table row with observation text and remarks
- `freeform`: group text by headings and paragraph blocks

Chunking details:

- table rows become fine-grained chunks with a location-aware context prefix
- page header/background text becomes its own chunk
- image appendix blocks become chunks with `image_b64` payloads
- long texts are split into overlapping segments using `TARGET_CHUNK_TOKENS = 400` and `OVERLAP_TOKENS = 50`
- chunk IDs are generated deterministically from the report stem, page number, and section index

When `eval_mode=True`, the chunker optionally aligns generated chunk IDs to a golden dataset mapping for evaluation.

### 4. Embeddings

The embedding manager is implemented in `src/isr_gpt/embedding/manager.py`.

Key behavior:

- loads a SentenceTransformer model from local path or Hugging Face
- moves the model to GPU if `torch.cuda.is_available()`
- supports `torch_dtype` override for lower precision
- adds a model-specific `query_prefix` to queries
- normalizes embeddings if configured

Embeddings are computed for document chunks before storing, and for query text during retrieval.

### 5. Vector storage

`VectorStore` in `src/isr_gpt/vectorstore/store.py` wraps ChromaDB.

It provides:

- persistent client initialization with `chromadb.PersistentClient`
- collection creation and naming sanitization
- batched upsert of ids, documents, metadatas, and embeddings
- search by embedding with optional metadata filters
- collection statistics and delete operations

Metadata stored per chunk includes:

- `chunk_id`
- `file_name`
- `file_path`
- `location`
- `page_num`
- `section`
- page geometry and rotation
- `has_image`
- bounding box coordinates
- `date_current` / `date_reference` if parsed from file name

### 6. Query analysis

`QueryAnalyzer` in `src/isr_gpt/retrieval/filters.py` analyzes the input query and extracts:

- known locations from `KNOWN_LOCATIONS`
- date ranges from natural date expressions
- query class: `simple`, `cross_location`, `exhaustive`, `temporal`, `analytical`, or `complex`

This intent is used to select retrieval behavior and filters.

### 7. Retrieval

`Retriever` in `src/isr_gpt/retrieval/retriever.py` is the pipeline connecting query to chunk search.

Steps:

1. Analyze the query with `QueryAnalyzer`
2. Build ChromaDB metadata filters from locations and dates
3. Choose a collection name (global or per-thread)
4. Determine search params based on query class
5. Embed the user query
6. Query ChromaDB for `initial_top_k` candidates
7. Optionally run query expansion for `complex` and `exhaustive` queries
8. Rerank the candidate chunks

Special behavior:

- `cross_location` queries retrieve separately for each location and merge results
- `exhaustive` queries may retrieve a large candidate pool and return more reranked chunks
- explicit `filters` can override inferred metadata filters

### 8. Reranking

`Reranker` in `src/isr_gpt/retrieval/reranker.py` performs a second-stage reordering with a cross-encoder.

It:

- loads `sentence_transformers.CrossEncoder`
- scores each query-chunk pair
- sorts chunks by cross-encoder score
- optionally applies diversity across source reports to avoid result collapse

The diversity selection chooses one top chunk per report in round-robin order, then fills remaining slots.

### 9. Generation

`GenerationManager` in `src/isr_gpt/generation/manager.py` handles LLM inference.

Important points:

- Loads GGUF models via `llama_cpp.Llama`
- defers LLM load until first generation when possible
- constructs chat messages from system prompt and user prompt
- calls `create_chat_completion` for normal or streaming responses
- classifies chunk citations by regex in `response_parser.py`

Prompt construction uses `src/isr_gpt/generation/prompt_templates.py`.

The user prompt includes:

- `SOURCE EXCERPTS:` section with chunk IDs and metadata
- `QUESTION: <query>`

The system prompt instructs the model to answer from provided excerpts and cite chunk IDs.

## Database Schema

The SQLite database is initialized in `src/isr_gpt/db/database.py`.

### Tables

- `users`
  - `id`, `username`, `password_hash`, `created_at`
  - seeded default user: `rakshit@galaxeye.space` / `GalaxEye@01`

- `threads`
  - `id`, `user_id`, `title`, `folder_path`, `collection_name`, `source_label`, `created_at`, `updated_at`

- `messages`
  - `id`, `thread_id`, `role`, `content`, `citations_json`, `created_at`

- `attachments`
  - `id`, `thread_id`, `file_name`, `file_path`, `created_at`, `last_ingested_at`, `file_mtime`

- `prompt_configs`
  - `id`, `user_id`, `name`, `prompt_text`, `is_active`, `created_at`, `updated_at`

- `canvases`
  - `id`, `user_id`, `thread_id`, `name`, `created_at`, `updated_at`

- `canvas_items`
  - `id`, `canvas_id`, `content`, `source_thread_id`, `source_thread_title`, `sort_order`, `created_at`

### Database behavior

- Enables WAL journaling and foreign key enforcement
- Applies migration helpers for schema evolution
- Uses `get_conn()` context manager for SQLite connections
- Automatically adds columns if needed during init

## API Endpoints

### Application status

- `GET /api/v1/status`
  - Reports GPU availability, VRAM usage, model load state, indexed chunk counts, and collections

### Authentication

- `POST /api/v1/auth/login`
  - accepts `username` and `password`
  - returns JWT bearer token

- `GET /api/v1/auth/me`
  - returns authenticated user info

### Configuration

- `GET /api/v1/config`
  - returns active and available embedding/LLM models

- `PUT /api/v1/config`
  - switch active embedding or LLM models at runtime

### Ingestion

- `POST /api/v1/ingest`
  - ingest a single PDF into the global collection

- `POST /api/v1/ingest/batch`
  - ingest a directory of PDFs into the global collection

### Query

- `POST /api/v1/query`
  - retrieve and generate an answer from the global collection

### Thread management

- `GET /api/v1/threads`
- `POST /api/v1/threads`
- `GET /api/v1/threads/{thread_id}`
- `PATCH /api/v1/threads/{thread_id}`
- `DELETE /api/v1/threads/{thread_id}`

### Thread-scoped ingestion and query

- `POST /api/v1/threads/{thread_id}/ingest`
  - ingest a PDF or directory into a thread-specific vector collection

- `POST /api/v1/threads/{thread_id}/upload`
  - upload and ingest PDF files into a thread

- `POST /api/v1/threads/{thread_id}/query`
  - query only the thread-scoped collection
  - automatically re-ingests new/modified files from the thread folder before running

### Prompt configuration

- `GET /api/v1/prompt-configs`
- `POST /api/v1/prompt-configs`
- `PATCH /api/v1/prompt-configs/{config_id}`
- `DELETE /api/v1/prompt-configs/{config_id}`
- `POST /api/v1/prompt-configs/{config_id}/activate`
- `POST /api/v1/prompt-configs/{config_id}/deactivate`

Active prompt configs are prepended to thread queries when enabled.

## Query and retrieval flow

When a query arrives via `/api/v1/query` or `/api/v1/threads/{thread_id}/query`:

1. Query analysis extracts locations, dates, and classifies the query
2. Query metadata filters are built from extracted locations/dates
3. The active embedding model encodes the query with optional `query_prefix`
4. ChromaDB returns candidate chunks by embedding similarity
5. Query expansion may add supplementary candidates for exhaustive or complex queries
6. The reranker rescoring model reorders results and optionally balances across source files
7. The LLM builds a prompt from the system prompt and selected chunk excerpts
8. The LLM returns an answer with chunk citations
9. The answer is returned together with citation metadata and latency

## Models and techniques used

### Embeddings

The system is built to use sentence-transformer models for dense retrieval.

The default active embedding is:

- `qwen3-embedding-4b`

Other supported models include:

- `qwen3-embedding-0.6b`
- `qwen3-embedding-8b`
- `bge-large-en-v1.5`
- `bge-base-en-v1.5`
- `nomic-embed-text-v1.5`
- `embeddinggemma-300m`
- `nv-embed-v2`

The embedding manager will:

- try loading from the local model path
- fall back to Hugging Face download if path is missing
- use GPU if available
- normalize vectors when configured
- add model-specific query prefixes for better retrieval

### Reranker

The reranker uses a cross-encoder from `sentence_transformers.CrossEncoder`.

The active reranker is:

- `ms-marco-MiniLM-L-6-v2`

The reranker re-scores query-chunk text pairs and supports diversity-aware selection across source reports.

### LLMs

The LLM component is built for local GGUF inference using `llama-cpp-python`.

Available models include:

- `gemma-4-e4b-it-q4` (default)
- `gemma-4-e4b-it`
- `gemma-4-e4b-it-q4-8gb`
- `gemma-4-27b-a4b-it`
- `gemma-4-27b-a4b-it-32gb`
- `phi-3.5-mini-instruct`
- `llama-3.1-8b-instruct`
- `qwen2.5-7b-instruct`

The generation manager configures `llama_cpp.Llama` with the model-specific parameters from `models.yaml`, including GPU layer allocation, batch sizes, flash attention, and KV cache quantization.

### Prompting

The system prompt is loaded from `config/system_prompt.txt` and instructs the model to:

- only use provided source excerpts
- cite chunk IDs in square brackets
- preserve qualifiers like "likely" or "probable"
- avoid hallucinating
- structure answers clearly

The user prompt includes a formatted excerpt list and the actual query.

## Thread collections and folders

Threads are user-scoped conversation contexts stored in SQLite.

Each thread can have:

- an associated folder path
- a dedicated ChromaDB collection name
- attachments tracked in the DB
- messages and citations stored for conversation history
- active prompt configs applied to queries

Thread ingestion uses a deterministic collection name based on thread UUID and embedding model. This keeps thread data isolated from the global collection.

## Database and auth

The app seeds a default user in `src/isr_gpt/db/database.py` using bcrypt.

Authentication is implemented in `src/isr_gpt/api/routes_auth.py` using JWT bearer tokens.

Auth-protected endpoints include thread management, prompt configs, and thread ingestion/query.

## Important implementation details

### Chunk IDs and citation tracking

Chunks are identified by deterministic IDs based on report stem, page, and section. This allows citation extraction from the LLM answer using a regex in `src/isr_gpt/generation/response_parser.py`.

### Structured vs freeform chunking

- structured IMINT tables become one chunk per observation row
- freeform text is grouped by heading and split into ~400-word chunks
- overlapping chunks preserve retrieval coverage for boundary content

### Query intent and filters

`QueryAnalyzer` uses regex-based extraction and keyword matching for:

- location filters
- date range filters
- query classification
- query expansion terms

The retriever builds ChromaDB `where` clauses from location and date conditions.

### Auto re-ingest for threads

The thread query endpoint can automatically detect new or modified PDFs in a thread folder. It re-ingests them into the thread's collection before answering.

### Collection naming

Global collections are named from the active embedding model.

Thread collections use a sanitized name with a short thread prefix so per-thread data stays isolated.

## File references

- `src/isr_gpt/main.py` - server bootstrapping and router registration
- `src/isr_gpt/config/settings.py` - runtime config and model registry
- `src/isr_gpt/db/database.py` - SQLite initialization and migrations
- `src/isr_gpt/ingestion/pdf_parser.py` - PDF parse logic
- `src/isr_gpt/ingestion/chunker.py` - chunk generation logic
- `src/isr_gpt/embedding/manager.py` - embedding infrastructure
- `src/isr_gpt/vectorstore/store.py` - ChromaDB wrapper
- `src/isr_gpt/retrieval/filters.py` - query analysis and intent extraction
- `src/isr_gpt/retrieval/retriever.py` - retrieval pipeline
- `src/isr_gpt/retrieval/reranker.py` - reranking pipeline
- `src/isr_gpt/generation/manager.py` - LLM generation lifecycle
- `src/isr_gpt/generation/prompt_templates.py` - prompt composition
- `src/isr_gpt/generation/response_parser.py` - citation extraction
- `src/isr_gpt/api/routes_ingest.py` - ingestion API
- `src/isr_gpt/api/routes_query.py` - global query API
- `src/isr_gpt/api/routes_thread_ingest.py` - thread ingestion API
- `src/isr_gpt/api/routes_thread_query.py` - thread query API
- `src/isr_gpt/api/routes_auth.py` - authentication API
- `src/isr_gpt/api/routes_config.py` - runtime model switching API
- `src/isr_gpt/api/routes_prompt_configs.py` - prompt config CRUD API

## How to use

### Ingest a report

POST to `/api/v1/ingest` with JSON:

```json
{
  "pdf_path": "C:/path/to/report.pdf",
  "location": "Hotan"
}
```

### Query the global collection

POST to `/api/v1/query` with JSON:

```json
{
  "query": "What changes were observed in Hotan in November 2023?",
  "top_k": 12
}
```

### Create a thread and ingest into it

1. `POST /api/v1/threads` to create the thread
2. `POST /api/v1/threads/{thread_id}/ingest` or `/upload`
3. `POST /api/v1/threads/{thread_id}/query`

### Switch models at runtime

PUT `/api/v1/config` with JSON:

```json
{
  "active_embedding_model": "bge-large-en-v1.5",
  "active_llm_model": "gemma-4-e4b-it-q4"
}
```

## Notes and guidelines

- `llama-cpp-python` is required for the LLM backend. Install it with CUDA support if you need GPU acceleration.
- The active embedding model determines the ChromaDB global collection name.
- Thread-scoped ingestion keeps thread data isolated, but shared global ingestion writes to the default collection.
- The system prompt and prompt configs are a primary mechanism for controlling model behavior.
- Use prompt configs to add domain-specific instructions or stricter citation policies.

## Troubleshooting

- If PDF parsing misses table rows, check that `PyMuPDF` is detecting tables or use the fallback heuristics.
- If embeddings fail to load, verify the local model path and `hf_name` in `config/models.yaml`.
- If the LLM fails to load, ensure the GGUF model file exists and `llama-cpp-python` is installed.
- If thread queries return zero results, verify the thread collection has been created and populated.

## Extending the system

To extend the system:

- add new query intent categories in `src/isr_gpt/retrieval/filters.py`
- add new chunk metadata in `src/isr_gpt/vectorstore/store.py`
- add a new reranker model in `config/models.yaml`
- add a new LLM model in `config/models.yaml`
- add additional API routes under `src/isr_gpt/api/`
- improve PDF parsing rules in `src/isr_gpt/ingestion/pdf_parser.py`

## Security note

JWT secret and seeded credentials are currently hardcoded in `src/isr_gpt/api/routes_auth.py` and `src/isr_gpt/db/database.py`. For production, move secrets into secure environment configuration.

---

This README is intended to document the entire ISR-GPT backend architecture, processing pipeline, model usage, schema design, and API wiring.
