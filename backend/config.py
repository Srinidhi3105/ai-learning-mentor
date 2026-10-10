import os
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        BASE_DIR / "data"
    )
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODELS
# ============================================================

GENERATION_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


# ============================================================
# RETRIEVAL
# ============================================================

VECTOR_CANDIDATES = 10
BM25_CANDIDATES = 10
RETRIEVAL_K = 10
FINAL_TOP_K = 3

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

RERANK_THRESHOLD = 4.0


# ============================================================
# UPLOAD
# ============================================================

MAX_UPLOAD_SIZE = 25 * 1024 * 1024


# ============================================================
# STORAGE
# ============================================================

DATABASE_FILE = str(
    DATA_DIR / "learning_mentor.db"
)

CHUNKS_FILE = str(
    DATA_DIR / "pdf_chunks.json"
)

FAISS_FILE = str(
    DATA_DIR / "pdf_faiss.index"
)

LOG_FOLDER = str(
    DATA_DIR / "logs"
)