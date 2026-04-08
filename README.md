# Local AI Document Assistant

> A fully offline, agentic AI system that reads, understands, and answers questions about your private document collections — with zero data leaving your machine.

![Status](https://img.shields.io/badge/Status-Active%20Development-green)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![Offline](https://img.shields.io/badge/100%25-Offline-red)

---

## What Is This?

The Local AI Document Assistant lets you build a private, intelligent knowledge base from your own documents. Point it at a folder of PDFs, Word files, emails, spreadsheets, images, or scanned pages — and ask questions in plain language. It finds the right information, cites the source, and reasons across multiple documents to give you a complete answer.

Everything runs on your own hardware. No subscription. No cloud. No data ever leaves your machine.

> **Agentic, not just search.** This is not a simple document search tool. The system uses AI agents that plan, retrieve iteratively, cross-reference sources, flag contradictions, and synthesize answers — the same way a knowledgeable colleague would, rather than a keyword search engine.

### Works for Any Domain

| Domain | Domain | Domain |
|--------|--------|--------|
| ⚡ SCADA / Industrial Systems | ⚖️ Legal Research | 🏥 Medical Protocols |
| 🔬 Research Papers | 📚 Training Manuals | 💼 Company Knowledge Base |
| 🧾 Financial Reports | 🏗️ Engineering Standards | 📧 Email Archives |
| 🎓 Course Materials | 📋 Compliance Documents | 🛠️ Technical Support |

---

## Key Features

- 🔒 **Fully Offline & Private** — No API keys. No cloud services. Every inference runs locally on your CPU or GPU. Your documents never leave your machine.
- 🗂️ **All Document Types** — PDF (digital + scanned OCR), Word, Excel, email (.eml), plain text, Markdown, CSV, and images — all ingested automatically.
- 🔍 **Hybrid Search** — Combines semantic vector search (ChromaDB) with BM25 keyword matching. Finds both conceptual answers and exact technical terms.
- 🤖 **Four Specialist Agents** — Router, Teacher, Troubleshooter, and Work Checker — each optimized for a different type of question.
- 📎 **Always Cites Sources** — Every answer includes the source document and page number. Contradictions between sources are flagged explicitly.
- 🧠 **Chain of Thought Reasoning** — Agents reason through the documents before answering, showing their work so you can follow the logic.
- 📊 **Confidence Scoring** — Every answer includes a 1-5 confidence score. Low-confidence answers (1-2) display a visible warning to verify before acting.
- 📁 **Multi-Project Workspaces** — Each topic gets its own isolated index, chat history, and configuration.
- 📂 **Bulk Folder Ingestion** — Point the system at any folder on your machine and ingest hundreds of documents at once via the web UI.
- 🖼️ **Vision Capability** — Submit photos or screenshots for live analysis using Gemma 4's native vision support.
- 🌐 **Browser Interface** — A custom dark-themed web UI accessible from any browser on your local network. No command line needed for end users.
- ⚙️ **One Config File** — Everything lives in a single `config.yaml`. Deploy on any OS or hardware by editing one file.

---

## The Four Agents

Every message is first classified by the Router Agent, which automatically sends it to the right specialist. You never need to choose — just ask your question naturally.

### 🟡 Agent 01 — Router
**Model:** `gemma4:e4b` (fast)

Reads every incoming message and classifies intent — teaching request, troubleshooting, work verification, or simple lookup. Routes to the correct specialist instantly. Uses the small, fast model to keep latency low.

### 🔵 Agent 02 — Teacher
**Model:** `gemma4:31b` (deep)

Explains procedures, concepts, and processes step by step. Calibrates explanation depth to your apparent technical level. Cites which document each step came from. Handles follow-up questions in context.

### 🔴 Agent 03 — Troubleshooter
**Model:** `gemma4:31b` (deep)

Diagnoses problems by retrieving relevant manuals AND past notes or emails simultaneously. Surfaces historical precedents. Flags conflicting information between sources. Suggests corrective actions with citations.

### 🟢 Agent 04 — Work Checker
**Model:** `gemma4:31b` (deep)

Compares what you did (or plan to do) against the official procedure. Flags steps missed, steps out of order, and potential risks. Always cites the source procedure. Explicitly states its own uncertainty.

---

## How It Works

### Document Ingestion Pipeline

Every document you add flows through a six-stage pipeline automatically:

```
Detect → Read → Chunk → Embed → Index → Record
  ↓        ↓      ↓       ↓        ↓       ↓
file    pdf/ocr  1000   nomic   Chroma  SQLite
type    /docx    chars  -embed  +BM25   metadata
```

1. **Detect** — Identifies file type from extension
2. **Read** — Extracts raw text page by page (OCR for scanned content)
3. **Chunk** — Splits text into 1000-character overlapping segments
4. **Embed** — Generates vectors via `nomic-embed-text`
5. **Index** — Builds ChromaDB semantic index + BM25 keyword index
6. **Record** — Logs document and chunks in SQLite

### Hybrid Retrieval

When you ask a question, both search engines run simultaneously:

```
Your Query
    ├── ChromaDB Semantic Search  →  Top 20 semantic matches
    └── BM25 Keyword Search       →  Top 20 keyword matches
                ↓
        Hybrid Combiner (70% semantic / 30% keyword)
                ↓
           Re-ranker → Top 5 high-quality chunks
                ↓
        Gemma 4 31B synthesizes answer with citations
```

**Why both?** Semantic search finds conceptually similar content but may miss exact technical abbreviations. Keyword search finds exact terms but misses conceptual questions. Together they cover both cases. The 70/30 weight is configurable in `config.yaml`.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| LLM (main) | Gemma 4 31B via Ollama | Teaching, troubleshooting, work checking |
| LLM (router) | Gemma 4 E4B via Ollama | Fast intent classification |
| Embeddings | nomic-embed-text | Offline vector generation |
| Vector DB | ChromaDB | Semantic similarity search |
| Keyword Search | BM25Plus (rank_bm25) | Exact technical term matching |
| Agents | LangGraph | Agent orchestration & state |
| OCR | Tesseract + pytesseract | Scanned PDF & image text extraction |
| PDF Reading | pymupdf | Digital PDF text extraction |
| Word Docs | python-docx | .docx processing |
| Spreadsheets | openpyxl | .xlsx processing |
| Metadata | SQLite | Document & chunk tracking |
| UI | Open WebUI (Docker) | Browser interface |
| API | FastAPI | Backend connecting UI to agents |
| Config | PyYAML | Portable configuration system |

---

## Hardware Requirements

| Hardware | Recommended Model | Notes |
|----------|------------------|-------|
| 12GB VRAM | Gemma 4 31B (Q4) | Best quality — reference build |
| 8GB VRAM | Gemma 4 E4B | Good balance of speed and quality |
| 6GB VRAM | Gemma 4 E2B | Edge model, still capable |
| CPU only | Gemma 4 E2B | Works, ~1–3 tokens/sec |
| Apple Silicon | Gemma 4 31B via MLX | Excellent performance on M-series |

**Reference build:** MINISFORUM UM760 mini PC (Ryzen 5 7640HS, 32GB RAM) with RTX 3060 eGPU (12GB VRAM) running Ubuntu Linux.

---

## Quick Start

### 1. Prerequisites

- [Ollama](https://ollama.com) installed and running
- Docker installed
- Python 3.11+
- Tesseract OCR: `sudo apt install tesseract-ocr`

### 2. Pull Required Models

```bash
# Main reasoning model
ollama pull gemma4:31b

# Fast router model
ollama pull gemma4:e4b

# Embedding model (required for search)
ollama pull nomic-embed-text
```

### 3. Clone and Configure

```bash
git clone https://github.com/billcw/test_agentic_AI_workflow.git
cd local-ai-doc-assistant

# Create your personal config from the template
cp config.example.yaml config.yaml
# Edit config.yaml with your paths and model choices
```

### 4. Set Up Python Environment

```bash
python3.11 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 5. Run Setup Verification

```bash
python setup.py
```

All five checks should pass before continuing.

### 6. Launch the API Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to `http://localhost:8000`.

### 7. Create Your First Project

1. Click **+ New** in the top bar
2. Enter a project name (e.g. `scada-ems`)
3. Click **Create**

### 8. Ingest Documents

**Single or multiple files:** Click **⬆ Upload Files** and select one or more files.

**Entire folder (recommended for bulk):** Type an absolute folder path into the **📁 Ingest folder** bar and click **Ingest Folder**. The server reads documents directly from disk — no uploading required.

See [docs/ADDING_DOCUMENTS.md](docs/ADDING_DOCUMENTS.md) for full details.

### 9. Ask Your First Question

Type a question and press Enter. The system routes it automatically:
- **TEACH** — step-by-step instructions with citations
- **TROUBLESHOOT** — structured diagnosis and corrective actions
- **CHECK** — verify work against official procedures
- **LOOKUP** — quick facts and definitions

Each answer shows the intent badge, a confidence score (1-5), and the source documents used.

---

## Folder Structure

```
local-ai-doc-assistant/
├── config.example.yaml     ← template committed to GitHub
├── config.yaml             ← YOUR settings (gitignored)
├── requirements.txt
├── setup.py                ← first-run verification script
├── src/
│   ├── config.py           ← loads config.yaml globally
│   ├── ingestion/          ← document processing pipeline
│   ├── storage/            ← ChromaDB, BM25, SQLite
│   ├── retrieval/          ← hybrid search + re-ranker
│   ├── agents/             ← four LangGraph agents
│   ├── memory/             ← per-project chat history
│   └── api/                ← FastAPI backend
├── workspaces/             ← YOUR data (gitignored)
│   └── your-project/
│       ├── vectors/        ← ChromaDB files
│       ├── bm25_index/     ← keyword index
│       ├── metadata.db     ← document tracker
│       └── chat_history.db
├── tests/
└── docs/
```

---

## Build Status

| Phase | Goal | Status |
|-------|------|--------|
| Phase 0 | Ollama + Gemma 4 + Open WebUI running | ✅ Complete |
| Phase 1 | Portable project skeleton + Git repo | ✅ Complete |
| Phase 2 | Document ingestion pipeline — all file types | ✅ Complete |
| Phase 3 | Hybrid retrieval — semantic + keyword + re-ranker | ✅ Complete |
| Phase 4 | Agentic layer — four LangGraph agents | ✅ Complete |
| Phase 5 | Team interface — FastAPI + custom web UI | ✅ Complete |
| Phase 6 | GitHub release — tests, docs, README complete | ✅ Complete |
| Post-6 | Chain of Thought reasoning + confidence scoring | ✅ Complete |
| Post-6 | Bulk folder ingestion via web UI | ✅ Complete |

---

## Design Principles

- 🔒 **Privacy First** — 100% offline. No telemetry, no cloud APIs, no subscriptions.
- 📦 **Portable by Design** — One `config.yaml` adapts the entire system to any OS, hardware, or drive layout.
- 🧩 **Modular Architecture** — Every component is independently testable and swappable via config.
- 📣 **Always Explainable** — Every answer cites its source. Contradictions are flagged, never silently resolved.
- 🚫 **No LangChain Monolith** — LangGraph for agents, direct Ollama calls for LLM, direct ChromaDB for vectors. No black-box abstraction layers.

---

## Configuration

The entire system is driven by a single `config.yaml` file. Copy `config.example.yaml` to `config.yaml` and edit your paths once. `config.yaml` is gitignored — only the example template is committed.

```yaml
paths:
  workspaces_root: "/home/yourname/local-ai-doc-assistant/workspaces"
  documents_root:  "/path/to/your/documents"
  tesseract_path:  "/usr/bin/tesseract"

models:
  llm:         "gemma4:31b"   # change to e4b for 8GB VRAM
  router_llm:  "gemma4:e4b"
  embeddings:  "nomic-embed-text"

retrieval:
  hybrid_weight: 0.7          # 0.7 = 70% semantic, 30% keyword
  chunk_size:    1000
  chunk_overlap: 200
```

---

## License

Released under the **Apache 2.0 License** — free to use, modify, and distribute for personal or commercial purposes.

Built by **Bill Wenz**, April 2026.

Contributions, bug reports, and hardware compatibility notes welcome via [GitHub Issues](https://github.com/billcw/test_agentic_AI_workflow/issues).
