# Hardware Guide

## Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16GB | 32GB |
| Storage | 50GB free | 200GB+ free |
| OS | Ubuntu 20.04, Windows 10, macOS 12 | Ubuntu 22.04+ |
| Python | 3.11 | 3.11 |

GPU is strongly recommended but not required. CPU-only is supported.

---

## Model Selection by Hardware

| VRAM / RAM | Recommended Model | Speed | Quality |
|------------|------------------|-------|---------|
| 12GB VRAM | gemma4:31b | Medium | Best |
| 8GB VRAM | gemma4:e4b | Fast | Good |
| 6GB VRAM | gemma4:e2b | Fast | Capable |
| CPU only (16GB+ RAM) | gemma4:e2b | Slow (1-3 tok/sec) | Capable |
| Apple Silicon M1/M2/M3 | gemma4:31b via MLX | Fast | Best |

To change the model, edit `config.yaml`:
```yaml
models:
  llm: "gemma4:e4b"   # Change this to match your hardware
```

---

## Reference Build

**Bill's build — the system was developed and tested on this hardware:**

| Component | Specification |
|-----------|--------------|
| Mini PC | MINISFORUM UM760 — Ryzen 5 7640HS, 32GB RAM |
| GPU | NVIDIA RTX 3060 in eGPU enclosure — 12GB VRAM |
| Storage | Internal SSD + external drives |
| OS | Ubuntu Linux |
| Model | gemma4:31b (Q4 quantization via Ollama) |

---

## GPU Setup (Linux)

If using an eGPU or external GPU enclosure, Ollama may not detect it
after a reboot. Run this after every restart if needed:
```bash
sudo systemctl restart ollama
```

Verify the GPU is active:
```bash
ollama run gemma4:e4b "hello"
# Watch GPU usage with: nvidia-smi
```

---

## Apple Silicon

Ollama runs natively on Apple Silicon via Metal. No additional setup needed.
gemma4:31b runs well on M2/M3 machines with 16GB+ unified memory.
```bash
# Install Ollama for Mac from https://ollama.com
ollama pull gemma4:31b
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

---

## Windows

All components support Windows. Key differences:

- Tesseract path: `C:/Program Files/Tesseract-OCR/tesseract.exe`
- Activate virtualenv: `venv\Scripts\activate`
- Paths in config.yaml use forward slashes or escaped backslashes
```yaml
paths:
  workspaces_root: "C:/Users/YOUR_USERNAME/local-ai-doc-assistant/workspaces"
  tesseract_path:  "C:/Program Files/Tesseract-OCR/tesseract.exe"
  temp_dir:        "C:/Temp/doc_assistant"
```

---

## Storage Recommendations

- **Models** (Ollama): gemma4:31b requires ~20GB. Store on your fastest drive.
- **Workspaces**: ChromaDB vector indexes grow with document count. Allow 1-2GB per 1000 documents.
- **Documents**: Keep originals anywhere. The config `documents_root` points to them.
