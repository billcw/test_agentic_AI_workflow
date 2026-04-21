"""
api/server.py - FastAPI backend for the Local AI Document Assistant.

To run the server:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
import requests as http_requests
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.config import INTERFACE, PATHS, OLLAMA
from src.agents.orchestrator import run_agent
from src.ingestion.pipeline import ingest_file, ingest_directory
from src.memory.history import (
    save_turn, get_recent_history,
    get_session_history, list_sessions
)
from src.projects.manager import (
    list_projects, create_project,
    project_exists, delete_project
)
from src.tools.file_organizer import classify_files, execute_plan, filter_images


# --- App Setup ---

app = FastAPI(
    title="Local AI Document Assistant",
    description="Offline agentic RAG system for private document collections.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the web UI) from src/api/static/
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# --- Request / Response Models ---

class QueryRequest(BaseModel):
    project_name: str
    query: str
    session_id: Optional[str] = None
    router_model: Optional[str] = None
    reasoning_model: Optional[str] = None
    top_k: Optional[int] = None
    top_k_final: Optional[int] = None
    hybrid_weight: Optional[float] = None
    email_max_chars: Optional[int] = None
    doc_max_chars: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list
    chunks_used: int
    confidence: int
    session_id: str


class CreateProjectRequest(BaseModel):
    project_name: str


class IngestFolderRequest(BaseModel):
    project_name: str
    folder_path: str
    force: bool = False


class OrganizeFolderRequest(BaseModel):
    folder_path: str
    custom_instructions: str = ""


class ExecutePlanRequest(BaseModel):
    plan: dict
class ImageFilterRequest(BaseModel):
    source_folder: str
    query: str
    destination_folder: str


# --- Image Filter Endpoint ---
@app.post("/filter_images")
def filter_images_endpoint(request: ImageFilterRequest):
    """
    Scan a source folder for images, describe each one with vision,
    and move those matching the query to the destination folder.
    """
    try:
        result = filter_images(
            source_folder=request.source_folder,
            query=request.query,
            destination_folder=request.destination_folder,
        )
        return result
    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Serve Web UI ---

@app.get("/")
def serve_ui():
    """Serve the main web UI."""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404,
                            detail="UI not found. index.html missing.")
    return FileResponse(str(index_path))


# --- Health Check ---

@app.get("/health")
def health_check():
    """Simple liveness check."""
    return {"status": "ok", "service": "local-ai-doc-assistant"}


# --- Models Endpoint ---

@app.get("/models")
def get_models():
    """
    Return the list of models currently available in Ollama.
    The UI uses this to populate the model selection dropdowns.
    """
    try:
        response = http_requests.get(
            f"{OLLAMA['base_url']}/api/tags",
            timeout=10
        )
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = sorted([m["name"] for m in models])
        return {"models": model_names}
    except Exception as e:
        return {"models": [], "error": str(e)}


# --- Query Endpoint ---

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main query endpoint. Routes the user's message through the full
    agentic pipeline and returns the answer with citations.
    """
    if not project_exists(request.project_name):
        raise HTTPException(
            status_code=404,
            detail=f"Project '{request.project_name}' not found."
        )

    session_id = request.session_id or str(uuid.uuid4())

    chat_history = get_recent_history(
        request.project_name, session_id, max_turns=6
    )

    result = run_agent(
        project_name=request.project_name,
        query=request.query,
        chat_history=chat_history,
        router_model=request.router_model,
        reasoning_model=request.reasoning_model,
        top_k=request.top_k,
        top_k_final=request.top_k_final,
        hybrid_weight=request.hybrid_weight,
        email_max_chars=request.email_max_chars,
        doc_max_chars=request.doc_max_chars
    )

    save_turn(
        project_name=request.project_name,
        session_id=session_id,
        user_message=request.query,
        assistant_reply=result["answer"],
        intent=result["intent"],
        sources=result["sources"]
    )

    return QueryResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=result["sources"],
        chunks_used=result["chunks_used"],
        confidence=result.get("confidence", 3),
        session_id=session_id
    )


# --- Document Ingestion Endpoints ---

@app.post("/ingest")
async def ingest_document(
    project_name: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and ingest a single document into a project workspace."""
    if not project_exists(project_name):
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_name}' not found."
        )

    temp_dir = Path(PATHS.get("temp_dir", "/tmp/doc_assistant"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        contents = await file.read()
        temp_path.write_bytes(contents)
        result = ingest_file(project_name, str(temp_path))
        return {
            "status": result["status"],
            "filename": file.filename,
            "chunks": result.get("chunks", 0),
            "message": result.get("message", "")
        }
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/ingest_folder")
def ingest_folder(request: IngestFolderRequest):
    """
    Ingest all supported documents from a folder path on the server.

    This is for bulk ingestion of documents already on the server machine.
    The folder_path must be an absolute path accessible to the server process.

    Why a server-side path instead of uploading files:
    When you have hundreds of documents already on disk (e.g. on an external
    drive), re-uploading them through the browser would be slow and pointless.
    This endpoint lets the server read them directly from disk.

    Returns a summary of what was ingested, skipped, and errored.
    """
    if not project_exists(request.project_name):
        raise HTTPException(
            status_code=404,
            detail=f"Project '{request.project_name}' not found."
        )

    folder = Path(request.folder_path)

    if not folder.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Folder not found: {request.folder_path}"
        )

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a directory: {request.folder_path}"
        )

    result = ingest_directory(
        project_name=request.project_name,
        directory=folder,
        force=request.force
    )

    return {
        "status": "complete",
        "folder": str(folder),
        "ingested": result["ingested"],
        "skipped": result["skipped"],
        "errors": result["errors"],
        "total": result["total"],
        "results": result["results"]
    }


# --- File Organizer Endpoints ---

@app.post("/organize_folder")
def organize_folder(request: OrganizeFolderRequest):
    """
    Scan a folder and return a proposed classification plan (dry-run).

    Uses gemma4:e4b to classify each file into a category. Categories
    are generated on the fly — no hardcoded list.

    This endpoint NEVER moves files. It only returns the proposed plan.
    The UI shows the plan to the user, who must confirm before execution.


    """


    try:
        plan = classify_files(
            folder_path=request.folder_path,
            custom_instructions=request.custom_instructions or None,
        )
        return plan

    except (FileNotFoundError, NotADirectoryError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    except ValueError as e:
        # Safety refusal — path inside project directory
        raise HTTPException(status_code=403, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/organize_folder/execute")
def organize_folder_execute(request: ExecutePlanRequest):
    """
    Execute a confirmed file organization plan.

    The plan must be the exact dict returned by POST /organize_folder.
    The UI sends it back here only after the user has reviewed and
    confirmed the proposed moves.

    Creates category subfolders and moves files into them.
    Skips files that no longer exist at their source path.
    Renames on collision (appends _1, _2, etc.) to prevent overwrites.
    """
    try:
        result = execute_plan(request.plan)
        return result

    except ValueError as e:
        # Safety refusal
        raise HTTPException(status_code=403, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Project Management Endpoints ---

@app.get("/projects")
def get_projects():
    """List all available project workspaces."""
    return {"projects": list_projects()}


@app.post("/projects")
def new_project(request: CreateProjectRequest):
    """Create a new project workspace."""
    result = create_project(request.project_name)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.delete("/projects/{project_name}")
def remove_project(project_name: str):
    """Delete a project and all its data. Irreversible."""
    result = delete_project(project_name)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result


# --- Chat History Endpoints ---

@app.get("/history/{project_name}/sessions")
def get_sessions(project_name: str):
    """List all conversation sessions for a project."""
    if not project_exists(project_name):
        raise HTTPException(status_code=404,
                            detail=f"Project '{project_name}' not found.")
    return {"sessions": list_sessions(project_name)}


@app.get("/history/{project_name}/{session_id}")
def get_history(project_name: str, session_id: str):
    """Retrieve the full conversation history for a session."""
    if not project_exists(project_name):
        raise HTTPException(status_code=404,
                            detail=f"Project '{project_name}' not found.")
    return {"history": get_session_history(project_name, session_id)}


# --- Entry Point ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=INTERFACE.get("api_port", 8000),
        reload=True
    )
