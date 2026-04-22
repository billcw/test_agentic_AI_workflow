"""
src/tools/file_organizer.py - Two-pass AI-driven file organizer.

Pass 1 (broad): Classify files into broad topic categories.
Pass 2 (fine):  Classify files within one category into sub-topics.

Both passes use the same classify_files() function — only the prompt
changes (broad vs. specific granularity).

SAFETY RULES (non-negotiable):
  - Will REFUSE any path inside the project directory.
  - dry_run=True (default) returns a proposed plan — no files are moved.
  - Files only move when execute_plan() is called with a confirmed plan.

Why gemma4:e4b for classification?
  Each file requires its own LLM call. The e4b model is fast and cheap
  per call — using the 31b model here would be extremely slow for large
  directories. Classification is a simple task; e4b is well-suited to it.
"""

import re
import base64
import shutil
import requests
from pathlib import Path
from typing import Optional

from src.config import OLLAMA, MODELS

# ── Project directory protection ─────────────────────────────────────────────

PROJECT_ROOT = Path("/home/bill/local-ai-doc-assistant").resolve()

PROTECTED_PATHS = [
    PROJECT_ROOT,
]


def _assert_safe_path(path: Path) -> None:
    """
    Raise ValueError if the path is inside (or equal to) any protected
    directory.

    Why: The file organizer moves files. If it were ever pointed at the
    project directory itself, it could destroy source code, config, or
    the vector database. This check is the last line of defense.
    """
    resolved = path.resolve()
    for protected in PROTECTED_PATHS:
        try:
            resolved.relative_to(protected)
            raise ValueError(
                f"REFUSED: '{resolved}' is inside the protected project "
                f"directory '{protected}'. The file organizer cannot operate "
                f"on the project directory or any of its subdirectories."
            )
        except ValueError as e:
            if "REFUSED" in str(e):
                raise
            continue


# ── File content extraction ───────────────────────────────────────────────────

def _extract_text(file_path: Path, max_chars: int = 2000) -> Optional[str]:
    """
    Extract a text preview from a file using the project's existing readers.

    Returns None if the file type is not supported or the reader fails —
    the caller will then classify by filename only.

    Why max_chars=2000?
      We only need enough content to classify the file. 2000 characters
      (roughly one page) is sufficient for reliable classification without
      making every LLM call slow.
    """
    suffix = file_path.suffix.lower()

    try:
        # Plain text variants — read directly
        if suffix in (".txt", ".md", ".csv", ".log", ".py", ".js", ".html"):
            text = file_path.read_text(errors="replace")
            return text[:max_chars]

        # PDF
        if suffix == ".pdf":
            from src.ingestion.pdf_reader import read_pdf
            chunks = read_pdf(str(file_path))
            combined = " ".join(c.get("text", "") for c in chunks)
            return combined[:max_chars]

        # Word documents
        if suffix in (".docx", ".doc"):
            from src.ingestion.docx_reader import read_docx
            chunks = read_docx(str(file_path))
            combined = " ".join(c.get("text", "") for c in chunks)
            return combined[:max_chars]

        # Email files — read_email handles both .eml and .msg internally
        if suffix in (".msg", ".eml"):
            from src.ingestion.email_reader import read_email
            chunks = read_email(str(file_path))
            combined = " ".join(c.get("text", "") for c in chunks)
            return combined[:max_chars]

        # Excel
        if suffix in (".xlsx", ".xls", ".xlsm"):
            from src.ingestion.excel_reader import read_excel
            chunks = read_excel(str(file_path))
            combined = " ".join(c.get("text", "") for c in chunks)
            return combined[:max_chars]

        # Images — use gemma4:e4b vision to describe content
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            return _describe_image(file_path)

        # Unknown type — fall through to filename-only classification
        return None

    except Exception:
        # If any reader fails, fall through to filename-only classification.
        # We don't want a single unreadable file to abort the whole batch.
        return None


# ── Image vision description ─────────────────────────────────────────────────

def _describe_image(file_path: Path, vision_model: Optional[str] = None) -> Optional[str]:
    """
    Use a vision model to generate a plain-language description of an image.

    vision_model: override the default router_llm for this call. Allows the
    UI to select a dedicated vision model without changing the global config.

    Returns a text description or None if the call fails.

    Why two-step (describe then classify)?
      Separating visual interpretation from category selection gives better
      results than asking the model to do both at once. The description feeds
      into _classify_file() exactly like PDF or email text does, so the
      category-accumulation logic works normally.

    Why base64?
      Ollama's vision API requires the image to be passed as a base64-encoded
      string in the "images" field of the /api/generate request body.
    """
    base_url = OLLAMA.get("base_url", "http://localhost:11434")
    model = vision_model or MODELS.get("router_llm", "gemma4:e4b")
    timeout = OLLAMA.get("timeout_seconds", 3600)

    try:
        image_data = base64.b64encode(file_path.read_bytes()).decode("utf-8")

        # Determine MIME type for the prompt context
        suffix = file_path.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

        prompt = (
            "Describe the content of this image in 1-3 sentences. "
            "Focus on what the image is ABOUT — the subject matter, "
            "topic, or category it belongs to. "
            "Do not describe colors, composition, or technical qualities. "
            "Just describe what you see in plain language."
        )

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 500,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        description = response.json().get("response", "").strip()

        if not description:
            return None

        return description[:2000]

    except Exception:
        # Fall through to filename-only classification if vision call fails
        return None


# ── LLM classification ────────────────────────────────────────────────────────

def _classify_file(
    filename: str,
    content_preview: Optional[str],
    existing_categories: list[str],
    custom_instructions: Optional[str] = None,
) -> str:
    """
    Ask gemma4:e4b to assign a category to one file.

    When content is available, the prompt leads with content and treats
    the filename as secondary context. This prevents the model from
    anchoring on uninformative filenames like 'misc.pdf' or 'doc1.docx'.

    custom_instructions: optional free-text guidance injected into the
    prompt before the classification task. Example:
      "prefer these categories: SCADA, CIP, Operations"

    Returns a plain string category name (e.g. "CIP Compliance").
    """
    base_url = OLLAMA.get("base_url", "http://localhost:11434")
    model = MODELS.get("router_llm", "gemma4:e4b")
    timeout = OLLAMA.get("timeout_seconds", 3600)

    # Build the existing categories hint
    if existing_categories:
        cats_block = (
            "Categories already created:\n"
            + "\n".join(f"  - {c}" for c in existing_categories)
            + "\n\nReuse one of these if the file fits. "
              "Only create a new category if none apply.\n\n"
        )
    else:
        cats_block = "No categories yet — you will create the first one.\n\n"

    # Build optional custom instructions block
    if custom_instructions and custom_instructions.strip():
        instructions_block = (
            "Additional instructions from the user:\n"
            + custom_instructions.strip()
            + "\n\n"
        )
    else:
        instructions_block = ""

    # Lead with content when available — filename is secondary
    has_content = bool(content_preview and content_preview.strip())

    if has_content:
        content_block = (
            f"File content (first ~2000 characters):\n"
            f"{content_preview.strip()[:1500]}\n\n"
            f"Filename (for additional context only): {filename}\n\n"
        )
    else:
        content_block = (
            f"Filename: {filename}\n"
            f"(File content could not be read — classify by filename only)\n\n"
        )

    task = (
        "You are classifying files for folder organization. "
        "Read the file content carefully and assign a SHORT, DESCRIPTIVE "
        "category name (2-5 words, title case, no punctuation). "
        "The category will become a folder name. "
        "Base your decision primarily on what the content is ABOUT, "
        "not on the filename.\n\n"
    )

    prompt = (
        f"{task}"
        f"{instructions_block}"
        f"{cats_block}"
        f"{content_block}"
        "Respond with ONLY the category name. "
        "No explanation. No punctuation. No quotes. Just the category name."
    )

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 20,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Clean up: strip quotes, newlines, extra whitespace
        category = re.sub(r'["\'\n\r]', '', raw).strip()

        # Sanity check
        if not category or len(category) > 60 or category.count(' ') > 6:
            return "Uncategorized"

        return category

    except Exception:
        return "Uncategorized"


# ── Main classification pass ──────────────────────────────────────────────────

def classify_files(
    folder_path: str,
    custom_instructions: Optional[str] = None,
) -> dict:
    """
    Scan a directory and classify every file using the LLM.

    custom_instructions: optional free-text guidance passed through to
    every _classify_file() call. Example:
      "prefer these categories: SCADA, CIP, Operations"

    Returns a plan dict:
    {
        "folder": "/abs/path/to/folder",
        "moves": [
            {
                "filename": "RTU_maintenance_guide.pdf",
                "source": "/abs/path/RTU_maintenance_guide.pdf",
                "category": "RTU Maintenance",
                "destination": "/abs/path/RTU Maintenance/RTU_maintenance_guide.pdf"
            },
            ...
        ],
        "categories": ["RTU Maintenance", "CIP Compliance", ...],
        "total_files": 47,
        "errors": [{"filename": "...", "error": "..."}]
    }

    This function NEVER moves files. It only builds the plan.
    Call execute_plan() to actually move files after user confirmation.
    """
    folder = Path(folder_path)

    _assert_safe_path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    # Non-recursive — top level only.
    # Why: the organizer creates subfolders. Recursing into its own output
    # on a re-run would double-classify. Users point it at subfolders
    # explicitly for Pass 2.
    files = [f for f in folder.iterdir() if f.is_file()]

    if not files:
        return {
            "folder": str(folder),
            "moves": [],
            "categories": [],
            "total_files": 0,
            "errors": []
        }

    moves = []
    errors = []
    existing_categories: list[str] = []

    for file_path in sorted(files):
        try:
            content_preview = _extract_text(file_path)
            category = _classify_file(
                filename=file_path.name,
                content_preview=content_preview,
                existing_categories=existing_categories,
                custom_instructions=custom_instructions,
            )

            if category not in existing_categories:
                existing_categories.append(category)

            destination = folder / category / file_path.name

            moves.append({
                "filename": file_path.name,
                "source": str(file_path),
                "category": category,
                "destination": str(destination),
            })

        except Exception as e:
            errors.append({
                "filename": file_path.name,
                "error": str(e)
            })

    return {
        "folder": str(folder),
        "moves": moves,
        "categories": sorted(existing_categories),
        "total_files": len(files),
        "errors": errors
    }


# ── Plan execution ────────────────────────────────────────────────────────────

def execute_plan(plan: dict) -> dict:
    """
    Execute a confirmed move plan returned by classify_files().

    Creates category subdirectories and moves files into them.
    Skips files that have already been moved (source no longer exists).
    Renames on collision (appends _1, _2, etc.) to prevent overwrites.

    Returns:
    {
        "moved": 42,
        "skipped": 2,
        "errors": [{"filename": "...", "error": "..."}]
    }
    """
    folder = Path(plan["folder"])

    # Re-run safety check — always
    _assert_safe_path(folder)

    moved = 0
    skipped = 0
    errors = []

    for move in plan.get("moves", []):
        source = Path(move["source"])
        destination = Path(move["destination"])

        if not source.exists():
            skipped += 1
            continue

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Avoid silent overwrites — rename on collision
            final_dest = destination
            counter = 1
            while final_dest.exists():
                stem = destination.stem
                suffix = destination.suffix
                final_dest = destination.parent / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(str(source), str(final_dest))
            moved += 1

        except Exception as e:
            errors.append({
                "filename": move["filename"],
                "error": str(e)
            })

    return {
        "moved": moved,
        "skipped": skipped,
        "errors": errors
    }


# ── Image filter ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def _image_matches_query(description: str, query: str, file_path: Optional[Path] = None) -> bool:
    """
    Determine whether an image matches a user query using a two-step approach.

    Step 1 — String pre-check:
      If any query word appears in the description (case-insensitive), return
      True immediately. This is fast, requires no LLM call, and handles the
      common case where the vision description already names the subject.
      Example: query "dog", description "a fluffy dog on a lawn" → True.

    Step 2 — Direct vision check (fallback):
      If the string pre-check fails AND file_path is provided, send the raw
      image to gemma4:e4b with a short, focused yes/no prompt.
      Why direct vision instead of description-based LLM matching?
        The description-matching path requires a long prompt, which causes
        e4b's thinking block to consume 150-250+ tokens before producing
        output — making num_predict unpredictable. A short direct vision
        prompt ("Does this image contain X?") keeps the thinking block small
        and fits reliably within num_predict=50.
      This also handles cases where the description frames the image
      artistically and obscures the actual subject (e.g. broccoli described
      primarily as a "whimsical arrangement with a birdhouse").

    Returns True if either step concludes the image matches.
    """
    base_url = OLLAMA.get("base_url", "http://localhost:11434")
    model = MODELS.get("router_llm", "gemma4:e4b")
    timeout = OLLAMA.get("timeout_seconds", 3600)

    # Step 1: string pre-check — any query word in description?
    query_words = [w.strip().lower() for w in query.split() if len(w.strip()) > 2]
    description_lower = description.lower()
    if any(word in description_lower for word in query_words):
        return True

    # Step 2: direct vision check on the raw image
    if file_path is None or not file_path.exists():
        return False

    try:
        image_data = base64.b64encode(file_path.read_bytes()).decode("utf-8")

        prompt = f"Does this image contain {query}? Respond with only YES or NO."

        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 50,
                }
            },
            timeout=timeout
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip().upper()
        return answer.startswith("YES")

    except Exception:
        return False


def filter_images(
    source_folder: str,
    query: str,
    destination_folder: str,
    vision_model: Optional[str] = None,
) -> dict:
    """
    Scan a folder for images, describe each one with vision, and move
    those that match the query to the destination folder.

    This is a filter+move tool — not a classifier. It does not create
    subfolders or build a plan for confirmation. Matches are moved
    immediately on the assumption that the user has reviewed the query.

    Why no dry-run / confirmation step here?
      The organizer needs confirmation because it proposes a folder
      structure the user must review. The image filter has a much simpler
      contract: "find images matching X and put them in Y". The query
      itself is the specification — the user knows what they asked for.

    Returns:
    {
        "source": "/abs/path/to/source",
        "destination": "/abs/path/to/destination",
        "query": "dogs outdoors",
        "total_images": 42,
        "matched": 7,
        "moved": 7,
        "skipped": 0,
        "results": [
            {
                "filename": "photo1.jpg",
                "description": "a dog running in a park",
                "matched": true,
                "moved": true,
                "destination": "/abs/path/to/destination/photo1.jpg"
            },
            ...
        ],
        "errors": [{"filename": "...", "error": "..."}]
    }
    """
    source = Path(source_folder)
    dest = Path(destination_folder)

    _assert_safe_path(source)
    _assert_safe_path(dest)

    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source_folder}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_folder}")
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # Collect image files only — top level
    image_files = [
        f for f in sorted(source.iterdir())
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    results = []
    errors = []
    matched_count = 0
    moved_count = 0
    skipped_count = 0

    for file_path in image_files:
        try:
            # Step 1: describe the image using vision
            description = _describe_image(file_path, vision_model=vision_model)

            if not description:
                # Vision call failed — skip this file
                errors.append({
                    "filename": file_path.name,
                    "error": "Vision description failed — skipped."
                })
                continue

            # Step 2: ask LLM if description matches query
            matched = _image_matches_query(description, query.strip(), file_path)

            result_entry = {
                "filename": file_path.name,
                "description": description,
                "matched": matched,
                "moved": False,
                "destination": None,
            }

            if matched:
                matched_count += 1

                # Step 3: move the file to destination
                dest.mkdir(parents=True, exist_ok=True)
                final_dest = dest / file_path.name

                # Collision handling
                counter = 1
                while final_dest.exists():
                    final_dest = dest / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1

                if file_path.exists():
                    shutil.move(str(file_path), str(final_dest))
                    moved_count += 1
                    result_entry["moved"] = True
                    result_entry["destination"] = str(final_dest)
                else:
                    skipped_count += 1

            results.append(result_entry)

        except Exception as e:
            errors.append({
                "filename": file_path.name,
                "error": str(e)
            })

    return {
        "source": str(source),
        "destination": str(dest),
        "query": query.strip(),
        "total_images": len(image_files),
        "matched": matched_count,
        "moved": moved_count,
        "skipped": skipped_count,
        "results": results,
        "errors": errors,
    }
