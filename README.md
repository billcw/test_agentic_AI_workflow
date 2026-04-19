# Local AI Document Assistant

> A fully offline, agentic AI system that reads, understands, and answers questions about your private document collections — with zero data leaving your machine.

![Status](https://img.shields.io/badge/Status-Active%20Development-green)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![Offline](https://img.shields.io/badge/100%25-Offline-red)

---

## What Is This?

The Local AI Document Assistant lets you build a private, intelligent knowledge base from your own documents. Point it at a folder of PDFs, Word files, emails, PST archives, spreadsheets, images, or scanned pages — and ask questions in plain language. It finds the right information, cites the source, and reasons across multiple documents to give you a complete answer.

Everything runs on your own hardware. No subscription. No cloud. No data ever leaves your machine.

> **Agentic, not just search.** This is not a simple document search tool. The system uses AI agents that plan, retrieve iteratively, cross-reference sources, flag contradictions, and synthesize answers — the same way a knowledgeable colleague would, rather than a keyword search engine.

> **Built on minimum viable hardware — intentionally.** A core goal of this project was to discover exactly how little hardware is needed to run a fully capable, production-quality agentic AI system. The reference build uses a consumer mini PC with an eGPU — not a server, not a workstation, not a cloud instance. If it runs well here, it runs anywhere.

---

## A Learning Project — By Design

This project was built as much to learn as to ship.

The author — an experienced SCADA/EMS operations professional — came into this project with deep domain expertise in industrial control systems and zero background in AI development, Python packaging, vector databases, or agentic architectures. Every component was built from scratch, one step at a time, with a deliberate focus on understanding the *why* behind every decision — not just getting it to work.

That constraint shaped everything:

- **Architecture decisions were justified, not just accepted.** When a proposed approach wasn't professionally sound, it was pushed back on and a better one was found.
- **Every technical concept was explained in plain language** before being implemented — analogies, examples, and the reasoning behind each choice are baked into the code comments throughout.
- **Mistakes were diagnostic opportunities.** Heredoc corruption, BM25Okapi IDF failures, ChromaDB `where=None` crashes, Gemma 4 extended thinking consuming all tokens — each one was understood before it was fixed.
- **The build was incremental and verified.** No step was built on top of an unconfirmed previous step. This produced a system where every layer is understood, not just assembled.

The result is a production-quality agentic RAG system built by someone learning AI development in real time — which means the code is intentionally readable, the comments explain intent not just mechanics, and the architecture reflects considered decisions rather than cargo-culted patterns.

If you are also learning — this codebase is meant to be readable by you.

---

### Works for Any Domain

| Domain | Domain | Domain |
|--------|--------|--------|
| ⚡ Industrial & Operations | ⚖️ Legal Research | 🏥 Medical Protocols |
| 🔬 Research Papers | 📚 Training Manuals | 💼 Company Knowledge Base |
| 🧾 Financial Reports | 🏗️ Engineering Standards | 📧 Email Archives |
| 🎓 Course Materials | 📋 Compliance Documents | 🛠️ Technical Support |

---

## Key Features

- 🔒 **Fully Offline & Private** — No API keys. No cloud services. Every inference runs locally on your CPU or GPU. Your documents never leave your machine.
- 🗂️ **All Document Types** — PDF (digital + scanned OCR), Word, Excel, PST/OST email archives, MSG files, plain text, Markdown, CSV, and images — all ingested automatically.
- 🔍 **Hybrid Search** — Combines semantic vector search (ChromaDB) with BM25 keyword matching. Finds both conceptual answers and exact technical terms.
- 🎯 **Scope-Aware Retrieval** — The router automatically detects whether a query targets emails, documents, or all sources and filters retrieval accordingly. No manual selection needed.
- 🤖 **Five Specialist Agents** — Router, Teacher, Troubleshooter, Work Checker, and Sentiment Analyst — each optimized for a different type of question.
- 📎 **Always Cites Sources** — Every answer includes the source document and page number. Contradictions between sources are flagged explicitly.
- 🧠 **Chain of Thought Reasoning** — Agents reason through the documents before answering, showing their work so you can follow the logic.
- 📊 **Confidence Scoring** — Every answer includes a 1-5 confidence score. Low-confidence answers (1-2) automatically trigger retrieval refinement before delivery.
- 🔄 **Iterative Refinement** — When confidence is low, the system automatically retries with a different retrieval strategy. One retry maximum to prevent loops.
- 🌈 **Source Diversity (MMR)** — Maximum Marginal Relevance prevents any single document from dominating results, ensuring answers draw from multiple sources.
- 🔁 **Multi-Turn Retrieval** — If the first retrieval pass returns only one source, a second pass fires automatically with a broadened query.
- 📁 **Multi-Project Workspaces** — Each topic gets its own isolated vector index, BM25 index, chat history, and configuration.
- 💬 **Chat History** — The system maintains per-project conversation context. Follow-up questions are understood in the context of prior turns.
- 📂 **Bulk Folder Ingestion** — Point the system at any folder on your machine and ingest hundreds of documents at once via the web UI.
- 🎛️ **Model Switching** — UI dropdowns allow switching router and reasoning models per query without restarting the server.
- ⚖️ **Hybrid Weight Slider** — Live UI control to adjust the semantic/keyword search balance (0-100%) per query without restarting.
- 📏 **Chunk Cap Controls** — Separate UI controls for email and document chunk character limits, allowing live tuning of context density without a server restart.
- 🖼️ **Vision Capability** — Submit photos or screenshots for live analysis using Gemma 4 native vision support.
- 🌐 **Browser Interface** — A custom dark-themed web UI accessible from any browser on your local network. No command line needed for end users.
- ⚙️ **One Config File** — Everything lives in a single `config.yaml`. Deploy on any OS or hardware by editing one file.

---

## The Five Agents

Every message is first classified by the Router Agent, which automatically sends it to the right specialist. You never need to choose — just ask your question naturally.

### 🟡 Agent 01 — Router
**Model:** `gemma4:e4b` (fast)

Classifies every incoming message by two dimensions simultaneously: intent (teach / troubleshoot / check / sentiment / lookup) and scope (email / document / all). Routes to the correct specialist with the correct retrieval filter applied automatically. Uses the small, fast model to keep classification latency low.

### 🔵 Agent 02 — Teacher
**Model:** `gemma4:31b` (deep)

Explains procedures, concepts, and processes step by step. Calibrates explanation depth to your apparent technical level. Cites which document each step came from. Handles follow-up questions in context using chat history.

### 🔴 Agent 03 — Troubleshooter
**Model:** `gemma4:31b` (deep)

Diagnoses problems by retrieving relevant manuals AND past notes or emails simultaneously. Surfaces historical precedents. Flags conflicting information between sources explicitly. Provides structured output: likely cause, diagnostic steps, corrective actions, escalation path.

### 🟢 Agent 04 — Work Checker
**Model:** `gemma4:31b` (deep)

Compares operator-reported actions against the official procedure. Flags steps missed, steps out of order, and potential risks. Always cites the source procedure. Explicitly states its own uncertainty and never approves work it cannot verify.

### 🟣 Agent 05 — Sentiment Analyst
**Model:** `gemma4:31b` (deep)

Analyzes emotional tone, urgency, and mood across emails and documents. Identifies frustrated, urgent, or critical communications and surfaces specific examples with citations. Automatically scoped to email sources when the query context indicates communications. Falls back to keyword analysis if the LLM call fails.

---

## How It Works

### Document Ingestion Pipeline

Every document you add flows through a six-stage pipeline automatically:

```
Detect → Read → Chunk → Embed → Index → Record
  |        |       |        |         |        |
file    pdf/ocr  1000    nomic    Chroma   SQLite
type    /docx    chars   -embed   +BM25    metadata
```

1. **Detect** — Identifies file type from extension — PDF, DOCX, XLSX, PST/OST, EML, MSG, images, text
2. **Read** — Extracts raw text page by page. Uses Tesseract OCR for scanned content and images
3. **Chunk** — Splits text into 1000-character overlapping segments (200-char overlap). Filters chunks under 100 chars
4. **Embed** — Generates semantic vectors via `nomic-embed-text` running locally through Ollama
5. **Index** — Adds vectors to ChromaDB with metadata. Rebuilds BM25Plus keyword index with all chunks
6. **Record** — Logs document and chunk count in SQLite. Prevents duplicate ingestion via chunk_id

### Hybrid Retrieval Pipeline (agentic-v2)

When you ask a question, the full pipeline runs:

```
Your Query
    |
    +-- Router: classify intent  (teach/troubleshoot/check/sentiment/lookup)
    +-- Router: classify scope   (email / document / all)
                    |
            Retrieval Node (scope-filtered)
                    |
    +-- ChromaDB Semantic Search  (scope-filtered)
    +-- BM25 Keyword Search       (scope-filtered)
                    |
        Hybrid Combiner (semantic x weight) + (keyword x (1-weight))
                    |
        MMR Diversity Pass (penalizes single-source dominance)
                    |
        Source Diversity Check -- if only 1 source, fire second pass
                    |
           Re-ranker -> Top N high-quality chunks
                    |
        Specialist Agent synthesizes answer with citations
                    |
        Confidence Check -- if score 1-2, retry with adjusted weight
                    |
        Critic Agent evaluates response quality
                    |
            Final answer delivered
```

---

## Quick Start

### Prerequisites

- Ubuntu Linux (or any Linux/macOS — Windows untested)
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- At least 8GB VRAM (12GB recommended for 31B model)
- Tesseract OCR: `sudo apt install tesseract-ocr`

### 1. Clone the Repository

```bash
git clone https://github.com/billcw/test_agentic_AI_workflow.git
cd test_agentic_AI_workflow
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Required Models

```bash
ollama pull gemma4:31b
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

### 5. Configure the System

```bash
cp config.example.yaml config.yaml
nano config.yaml   # edit paths to match your system
```

### 6. Run the Setup Verification

```bash
python setup.py
```

### 7. Start the Server

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

### 8. Open the Web UI

Navigate to `http://localhost:8000` in any browser. Create a project, ingest some documents, and start asking questions.

See [docs/ADDING_DOCUMENTS.md](docs/ADDING_DOCUMENTS.md) for full details on all ingestion options.

### 9. Ask Your First Question

Type a question and press Enter. The system routes it automatically:

| Intent | Example Query | What You Get |
|--------|--------------|-------------|
| **TEACH** | How do I configure X? | Step-by-step procedure with citations |
| **TROUBLESHOOT** | Why is alarm Y firing? | Structured diagnosis and corrective actions |
| **CHECK** | Check my work: I did steps 1, 2, 3 | Verification against official procedure |
| **SENTIMENT** | Show me urgent communications | Tone analysis with cited examples |
| **LOOKUP** | What does term Z mean? | Direct answer with source reference |

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
│   ├── config.py           ← loads config.yaml globally at import time
│   ├── ingestion/          ← document processing pipeline (all file types)
│   ├── storage/            ← ChromaDB, BM25Plus, SQLite
│   ├── retrieval/          ← hybrid search, MMR, multi-turn, re-ranker
│   ├── agents/             ← five LangGraph agents + orchestrator
│   ├── memory/             ← per-project chat history
│   └── api/                ← FastAPI backend + custom web UI
├── workspaces/             ← YOUR data (gitignored)
│   └── your-project/
│       ├── vectors/        ← ChromaDB files
│       ├── bm25_index/     ← keyword index
│       ├── metadata.db     ← document tracker
│       └── chat_history.db ← conversation history
├── tests/                  ← test suite (23 tests)
└── docs/
```

---

## Build Status

| Phase | Goal | Status |
|-------|------|--------|
| Phase 0 | Ollama + Gemma 4 running | ✅ Complete |
| Phase 1 | Portable project skeleton + Git repo | ✅ Complete |
| Phase 2 | Document ingestion pipeline — all file types | ✅ Complete |
| Phase 3 | Hybrid retrieval — semantic + keyword + re-ranker | ✅ Complete |
| Phase 4 | Agentic layer — LangGraph agents | ✅ Complete |
| Phase 5 | FastAPI + custom dark web UI | ✅ Complete |
| Phase 6 | GitHub release — tests, docs, README | ✅ Complete |
| Post-6 | Chain of Thought reasoning + confidence scoring | ✅ Complete |
| Post-6 | Bulk folder ingestion via web UI | ✅ Complete |
| Post-6 | PST/OST email archive ingestion | ✅ Complete |
| Post-6 | Model dropdown wiring in UI | ✅ Complete |
| agentic-v2 | MMR source diversity penalties | ✅ Complete |
| agentic-v2 | Multi-turn retrieval — second pass on low diversity | ✅ Complete |
| agentic-v2 | Iterative refinement — retry on low confidence | ✅ Complete |
| agentic-v2 | Sentiment agent — LLM-first with keyword fallback | ✅ Complete |
| agentic-v2 | Scope-aware retrieval — email/document/all filtering | ✅ Complete |
| agentic-v2 | Hybrid weight slider wired through full pipeline | ✅ Complete |
| agentic-v2 | UI-controllable chunk caps — email and document | ✅ Complete |
| agentic-v2 | MSG email file support via extract-msg | ✅ Complete |
| agentic-v2 | Critic agent — response evaluation framework | 🔄 In Progress |

---

## Design Principles

- 🔒 **Privacy First** — 100% offline. No telemetry, no cloud APIs, no subscriptions. Nothing leaves your hardware.
- 📦 **Portable by Design** — One `config.yaml` adapts the entire system to any OS, hardware, or drive layout.
- 🧩 **Modular Architecture** — Every component is independently testable and swappable via config. No monolithic dependencies.
- 📣 **Always Explainable** — Every answer cites its source document and page. Contradictions are flagged, never silently resolved.
- 🚫 **No LangChain Monolith** — LangGraph for agents, direct Ollama API calls for LLM, direct ChromaDB for vectors. No black-box abstraction layers.
- ⏱️ **Quality Over Speed** — Timeout set to 3600 seconds. Response time is not a design constraint. Accuracy is.
- 📖 **Readable by Design** — Code comments explain intent and reasoning, not just mechanics. Built to be understood, not just run.

---

## Configuration

The entire system is driven by a single `config.yaml` file. Copy `config.example.yaml` to `config.yaml` and edit your paths once. `config.yaml` is gitignored — only the example template is committed to GitHub.

> **Important:** `config.py` loads `config.yaml` once at import time. Config changes require a full server restart — `uvicorn --reload` alone will not pick up `config.yaml` changes.

```yaml
paths:
  workspaces_root: "/home/yourname/local-ai-doc-assistant/workspaces"
  documents_root:  "/path/to/your/documents"
  tesseract_path:  "/usr/bin/tesseract"
  temp_dir:        "/tmp/local-ai-doc-assistant"

ollama:
  base_url:         "http://localhost:11434"
  timeout_seconds:  3600

models:
  llm:          "gemma4:31b"   # change to e4b for 8GB VRAM
  router_llm:   "gemma4:e4b"
  embeddings:   "nomic-embed-text"
  num_predict:  3000             # max tokens per response

retrieval:
  hybrid_weight:  0.5   # 0.5 = 50% semantic, 50% keyword
  top_k:          15    # candidates per search pass
  top_k_final:    10    # chunks sent to agent after reranking
  chunk_size:     1000
  chunk_overlap:  200

interface:
  api_port:  8000
```

---

## Known Limitations and Technical Notes

- **Critic agent temporarily disabled** — Gemma 4 models return empty responses to structured evaluation prompts. The framework is architecturally complete and wired into the pipeline. Re-enabling requires an alternative prompting strategy or a model with reliable structured output.
- **Ollama API endpoint** — Must use `/api/generate`, not `/api/chat`. The `/api/chat` endpoint triggers Gemma 4 extended thinking mode which consumes all tokens on internal reasoning and returns empty content.
- **Hybrid weight sensitivity** — Semantic-heavy weighting (above 0.7) can systematically exclude document subsets when corpus types differ significantly. Default of 0.5 provides better balance across mixed PDF and email corpora. Use the UI slider to tune per query.
- **config.yaml loads at import time** — Changes to `config.yaml` require a full server restart. `uvicorn --reload` will not pick up config changes alone.
- **BM25Plus vs BM25Okapi** — BM25Okapi produces 0.0 IDF scores on small corpora. BM25Plus uses a lower-bound IDF formula that always produces positive scores and is used throughout.
- **ChromaDB batch limit** — ChromaDB `add()` has a maximum of approximately 5461 records per call. The pipeline uses batches of 500 to stay safely within limits.
- **Corrupt OST files** — Corrupt or partial OST archives will fail silently during ingestion. PST files are more reliably handled by libpff.
- **num_predict tuning** — 3000 tokens is tuned for the reference hardware. Increasing this on systems with insufficient VRAM will cause the 31B model to fall back to CPU.

---

## License

Released under the **Apache 2.0 License** — free to use, modify, and distribute for personal or commercial purposes.

Built by **Bill Wenz** with **Claude (Anthropic)** as AI pair programming partner, April 2026. Every component — architecture decisions, code, retrieval tuning, agent prompts, and documentation — was developed collaboratively in an iterative session-based workflow between Bill and Claude.

Contributions, bug reports, and hardware compatibility notes welcome via [GitHub Issues](https://github.com/billcw/test_agentic_AI_workflow/issues).
