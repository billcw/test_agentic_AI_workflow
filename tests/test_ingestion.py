"""
test_ingestion.py - Tests for the document ingestion pipeline.

Run with: pytest tests/test_ingestion.py -v
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="module")
def sample_text_file():
    """Create a temporary text file for testing."""
    content = (
        "Test Document for Ingestion Pipeline\n\n"
        "Section 1: Overview\n"
        "This is a test document used to verify the ingestion pipeline.\n"
        "The pipeline should detect this as a text file and index it.\n\n"
        "Section 2: Technical Terms\n"
        "The alrm_server process manages system alarms.\n"
        "The lpmd service must be restarted before alrm_server.\n"
        "Contact extension 4400 for critical failures.\n\n"
        "Section 3: Procedures\n"
        "Step 1: Log into the primary workstation.\n"
        "Step 2: Check the service status.\n"
        "Step 3: Restart the service if needed.\n"
        "Step 4: Verify the system is operating correctly.\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        return f.name


@pytest.fixture(scope="module")
def test_project():
    return "test-ingestion-pytest"


def test_detector_identifies_text_file(sample_text_file):
    from src.ingestion.detector import detect_file_type
    result = detect_file_type(sample_text_file)
    assert result == "text", f"Expected text, got {result}"


def test_text_reader_extracts_content(sample_text_file):
    from src.ingestion.text_reader import read_text
    result = read_text(sample_text_file)
    assert result is not None
    assert len(result) > 0
    assert "alrm_server" in result[0]["text"]


def test_chunker_splits_text():
    from src.ingestion.chunker import chunk_text
    long_text = "This is a sentence. " * 200
    chunks = chunk_text(long_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0


def test_full_ingestion_pipeline(sample_text_file, test_project):
    from src.ingestion.pipeline import ingest_file
    result = ingest_file(test_project, sample_text_file)
    assert result["status"] == "ingested", f"Ingestion failed: {result.get('message')}"
    assert result["chunks"] > 0


def test_semantic_search_returns_results(test_project):
    from src.retrieval.semantic_search import semantic_search
    results = semantic_search(test_project, "restart the service")
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]


def test_keyword_search_returns_results(test_project):
    from src.retrieval.keyword_search import keyword_search
    results = keyword_search(test_project, "alrm_server lpmd")
    assert len(results) > 0


def test_hybrid_search_returns_results(test_project):
    from src.retrieval.hybrid_search import hybrid_search
    results = hybrid_search(test_project, "restart alrm_server procedure")
    assert len(results) > 0
    assert "semantic_score" in results[0]
    assert "keyword_score" in results[0]
    assert "score" in results[0]


def test_metadata_db_records_document(test_project):
    from src.storage.metadata_db import get_project_stats
    stats = get_project_stats(test_project)
    assert stats["documents"] > 0
    assert stats["chunks"] > 0


def teardown_module(module):
    import shutil
    from src.config import PATHS
    test_project_path = Path(PATHS["workspaces_root"]) / "test-ingestion-pytest"
    if test_project_path.exists():
        shutil.rmtree(str(test_project_path))
