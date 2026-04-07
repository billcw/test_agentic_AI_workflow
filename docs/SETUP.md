# Setup Guide

## Prerequisites

Before you begin, ensure the following are installed on your system:

- **Python 3.11+** — `python3 --version`
- **Ollama** — [https://ollama.com](https://ollama.com)
- **Docker** — [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)
- **Tesseract OCR** — `sudo apt install tesseract-ocr` (Linux) or [installer](https://github.com/UB-Mannheim/tesseract/wiki) (Windows)
- **Git** — `git --version`

---

## Step 1 — Pull Required Ollama Models
```bash
ollama pull gemma4:31b        # Main reasoning model (12GB VRAM)
ollama pull gemma4:e4b        # Fast router model
ollama pull nomic-embed-text  # Embedding model (required)
```

See [HARDWARE.md](HARDWARE.md) for lower-VRAM model alternatives.

---

## Step 2 — Clone the Repository
```bash
git clone https://github.com/billcw/test_agentic_AI_workflow.git
cd local-ai-doc-assistant
```

---

## Step 3 — Create Your Config File
```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your paths:
```yaml
paths:
  workspaces_root: "/home/YOUR_USERNAME/local-ai-doc-assistant/workspaces"
  tesseract_path:  "/usr/bin/tesseract"
  temp_dir:        "/tmp/doc_assistant"

models:
  llm:         "gemma4:31b"
  router_llm:  "gemma4:e4b"
  embeddings:  "nomic-embed-text"
```

**Important:** `config.yaml` is gitignored and never committed. It stays on your machine only.

---

## Step 4 — Create Python Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## Step 5 — Run Setup Verification
```bash
python setup.py
```

All five checks must pass before continuing:
- ✅ Config file found
- ✅ Ollama reachable
- ✅ Required models available
- ✅ Tesseract installed
- ✅ Workspaces directory writable

---

## Step 6 — Start the API Server
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to `http://localhost:8000` to access the web UI.

---

## Step 7 — Create Your First Project

1. Click **+ New** in the top bar
2. Enter a project name (e.g. `scada-ems`)
3. Click **Create**
4. Select your new project from the dropdown

---

## Step 8 — Ingest Documents

Click **Upload Doc** and select any supported file:
- PDF (digital or scanned)
- Word (.docx)
- Excel (.xlsx)
- Email (.eml)
- Plain text, Markdown, CSV
- Images (.jpg, .png, .tiff)

See [ADDING_DOCUMENTS.md](ADDING_DOCUMENTS.md) for bulk ingestion.

---

## Step 9 — Ask Your First Question

Type a question in the chat input and press Enter or click Send.

The system will automatically route your question to the right agent:
- **TEACH** — step-by-step instructions
- **TROUBLESHOOT** — diagnosis and corrective actions
- **CHECK** — verify your work against official procedures
- **LOOKUP** — quick facts and definitions

---

## Running Tests
```bash
pytest tests/ -v
```

All 23 tests should pass. Tests require Ollama running with the router model available.

---

## Troubleshooting

**Ollama not reachable:** Ensure Ollama is running with `ollama list`. If using an eGPU, run `sudo systemctl restart ollama` after connecting the GPU.

**Model not found:** Run `ollama pull <model-name>` to download missing models.

**Import errors:** Ensure you activated the virtualenv with `source venv/bin/activate`.

**Port 8000 in use:** Change `api_port` in `config.yaml` and restart the server.
