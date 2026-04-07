"""
api/server.py - FastAPI backend for the Local AI Document Assistant.

This is the HTTP interface that sits between the browser/UI and the
agentic layer. It exposes clean REST endpoints so that Open WebUI
(or any HTTP client) can:
  - Send a query and get an agent response
  - Upload and ingest a document
  - List and create projects
  - Retrieve chat history

Analogy: Think of this as the front desk of an office. Requests come
in through the door (HTTP), get routed to the right department
(agents, ingestion, projects), and responses go back out.

To run the server:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import INTERFACE, PATHS
from src.agents.orchestrator import run_agent
from src.ingestion.pipeline import ingest_file
from src.memory.history import (
    save_turn, get_recent_history,
    get_session_history, list_sessions
)
from src.projects.manager import (
    list_projects, create_project,
    project_exists, delete_project
)


# --- App Setup ---

app = FastAPI(
    title="Local AI Document Assistant",
    description="Offline agentic RAG system for private document collections.",
    version="1.0.0"
)

# Allow Open WebUI (running on a different port) to call this API.
# CORS = Cross-Origin Resource Sharing. Without this, browsers block
# requests from one port to another as a security measure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Models ---
# Pydantic models define the shape of JSON request bodies.
# FastAPI validates incoming requests against these automatically.

class QueryRequest(BaseModel):
    project_name: str
    query: str
    session_id: Optional[str] = None   # If None, a new session is started


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list
    chunks_used: int
    session_id: str


class CreateProjectRequest(BaseModel):
    project_name: str


# --- Health Check ---

@app.get("/health")
def health_check():
    """Simple liveness check. Returns OK if the server is running."""
    return {"status": "ok", "service": "local-ai-doc-assistant"}


# --- Query Endpoint ---

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Main query endpoint. Routes the user's message through the full
    agentic pipeline and returns the answer with citations.

    If session_id is provided, chat history is loaded and passed to
    the agent so it has conversation context. If not provided, a new
    session ID is generated.
    """
    if not project_exists(request.project_name):
        raise HTTPException(
            status_code=404,
            detail=f"Project '{request.project_name}' not found. "
                   f"Create it first via POST /projects."
        )

    # Use provided session_id or generate a new one
    session_id = request.session_id or str(uuid.uuid4())

    # Load recent chat history for context
    chat_history = get_recent_history(
        request.project_name, session_id, max_turns=6
    )

    # Run the full agentic pipeline
    result = run_agent(
        project_name=request.project_name,
        query=request.query,
        chat_history=chat_history
    )

    # Persist this turn to history
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
        session_id=session_id
    )


# --- Document Ingestion Endpoint ---

@app.post("/ingest")
async def ingest_document(
    project_name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload and ingest a document into a project workspace.

    Accepts any supported file type (PDF, DOCX, TXT, etc.).
    Saves the file to a temp location, runs the ingestion pipeline,
    then returns the result.
    """
    if not project_exists(project_name):
        raise HTTPException(
            status_code=404,
            detail=f"Project '{project_name}' not found."
        )

    # Save uploaded file to temp directory
    temp_dir = Path(PATHS.get("temp_dir", "/tmp/doc_assistant"))
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / file.filename

    try:
        contents = await file.read()
        temp_path.write_bytes(contents)

        # Run ingestion pipeline
        result = ingest_file(project_name, str(temp_path))

        return {
            "status": result["status"],
            "filename": file.filename,
            "chunks": result.get("chunks", 0),
            "message": result.get("message", "")
        }
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


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
