from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field
from typing import Literal

from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

import pymupdf
import tiktoken
import numpy as np
import faiss
import sqlite3
import json
import os

from datetime import datetime


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

client = OpenAI()

app = FastAPI(
    title="AI Learning Mentor",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GENERATION_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

VECTOR_CANDIDATES = 10
BM25_CANDIDATES = 10
RETRIEVAL_K = 10
FINAL_TOP_K = 3

VECTOR_WEIGHT = 0.7
BM25_WEIGHT = 0.3

RERANK_THRESHOLD = 4.0

MAX_UPLOAD_SIZE = 25 * 1024 * 1024

DATABASE_FILE = "learning_mentor.db"


# ============================================================
# 2. API MODELS
# ============================================================

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class StoredHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    source_pages: list[int] = Field(
        default_factory=list
    )


class QuestionRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=200
    )

    question: str = Field(
        min_length=1,
        max_length=3000
    )


class QuestionResponse(BaseModel):
    answer: str
    source_pages: list[int]
    search_query: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int
    vectors: int


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[StoredHistoryMessage]


# ============================================================
# 3. RERANKER MODELS
# ============================================================

class RerankItem(BaseModel):
    chunk_index: int

    score: float = Field(
        ge=0,
        le=10
    )


class RerankResult(BaseModel):
    results: list[RerankItem]


# ============================================================
# 4. DATABASE
# ============================================================

def get_db_connection():

    return sqlite3.connect(
        DATABASE_FILE
    )


def init_database():

    connection = get_db_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            source_pages TEXT,
            created_at TEXT NOT NULL
        )
        """
    )


    # Check existing columns
    cursor.execute(
        "PRAGMA table_info(conversations)"
    )


    columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    # Migration for older database
    if "source_pages" not in columns:

        cursor.execute(
            """
            ALTER TABLE conversations
            ADD COLUMN source_pages TEXT
            """
        )


    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_conversations_session
        ON conversations(session_id)
        """
    )


    connection.commit()

    connection.close()


init_database()


# ============================================================
# 5. SAVE MESSAGE
# ============================================================

def save_message(
    session_id,
    role,
    content,
    source_pages=None
):

    if source_pages is None:
        source_pages = []


    connection = get_db_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        INSERT INTO conversations
        (
            session_id,
            role,
            content,
            source_pages,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            json.dumps(source_pages),
            datetime.now().isoformat()
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# 6. GET RECENT CHAT HISTORY FOR RAG
# ============================================================

def get_chat_history(
    session_id,
    limit=6
):

    connection = get_db_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT role, content
        FROM conversations
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (
            session_id,
            limit
        )
    )


    rows = cursor.fetchall()

    connection.close()


    rows.reverse()


    history = []


    for role, content in rows:

        history.append(
            HistoryMessage(
                role=role,
                content=content
            )
        )


    return history


# ============================================================
# 7. GET ALL CHAT HISTORY
# ============================================================

def get_all_chat_history(
    session_id
):

    connection = get_db_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT
            role,
            content,
            source_pages
        FROM conversations
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (
            session_id,
        )
    )


    rows = cursor.fetchall()

    connection.close()


    messages = []


    for (
        role,
        content,
        source_pages
    ) in rows:

        try:

            pages = (
                json.loads(source_pages)
                if source_pages
                else []
            )

        except json.JSONDecodeError:

            pages = []


        messages.append(
            StoredHistoryMessage(
                role=role,
                content=content,
                source_pages=pages
            )
        )


    return messages


# ============================================================
# 8. DELETE CHAT HISTORY
# ============================================================

def delete_chat_history(
    session_id
):

    connection = get_db_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        DELETE FROM conversations
        WHERE session_id = ?
        """,
        (
            session_id,
        )
    )


    connection.commit()

    connection.close()


# ============================================================
# 9. KNOWLEDGE BASE
# ============================================================

chunks = []

index = None

bm25 = None


# ============================================================
# 10. BUILD BM25
# ============================================================

def build_bm25(
    chunk_list
):

    tokenized_chunks = []


    for chunk in chunk_list:

        tokens = (
            chunk["text"]
            .lower()
            .split()
        )


        tokenized_chunks.append(
            tokens
        )


    return BM25Okapi(
        tokenized_chunks
    )


# ============================================================
# 11. LOAD EXISTING KNOWLEDGE BASE
# ============================================================

def load_existing_knowledge_base():

    global chunks
    global index
    global bm25


    if not os.path.exists(
        "pdf_chunks.json"
    ):
        return


    if not os.path.exists(
        "pdf_faiss.index"
    ):
        return


    try:

        with open(
            "pdf_chunks.json",
            "r",
            encoding="utf-8"
        ) as file:

            loaded_chunks = json.load(
                file
            )


        loaded_index = faiss.read_index(
            "pdf_faiss.index"
        )


        if (
            len(loaded_chunks)
            != loaded_index.ntotal
        ):

            print(
                "Chunk/vector count mismatch."
            )

            return


        chunks = loaded_chunks

        index = loaded_index

        bm25 = build_bm25(
            chunks
        )


        print(
            "Knowledge base loaded."
        )

        print(
            "Chunks:",
            len(chunks)
        )

        print(
            "Vectors:",
            index.ntotal
        )


    except Exception as error:

        print(
            "Knowledge base load failed:",
            error
        )


load_existing_knowledge_base()


# ============================================================
# 12. SINGLE EMBEDDING
# ============================================================

def get_embedding(
    text
):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )


    return response.data[0].embedding


# ============================================================
# 13. BATCH EMBEDDINGS
# ============================================================

def get_embeddings(
    texts,
    batch_size=64
):

    all_embeddings = []


    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:
            start + batch_size
        ]


        print(
            f"Embedding "
            f"{start + 1}-"
            f"{min(start + batch_size, len(texts))}"
            f"/{len(texts)}"
        )


        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )


        ordered = sorted(
            response.data,
            key=lambda item:
                item.index
        )


        for item in ordered:

            all_embeddings.append(
                item.embedding
            )


    return all_embeddings


# ============================================================
# 14. CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    if overlap >= chunk_size:

        raise ValueError(
            "Overlap must be smaller "
            "than chunk size."
        )


    encoding = tiktoken.get_encoding(
        "o200k_base"
    )


    tokens = encoding.encode(
        text
    )


    result = []

    start = 0


    while start < len(tokens):

        end = (
            start + chunk_size
        )


        chunk_tokens = tokens[
            start:end
        ]


        chunk = (
            encoding.decode(
                chunk_tokens
            )
            .strip()
        )


        if chunk:

            result.append(
                chunk
            )


        start = (
            end - overlap
        )


    return result


# ============================================================
# 15. BUILD PDF KNOWLEDGE BASE
# ============================================================

def build_pdf_knowledge_base(
    pdf_bytes,
    filename
):

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )


    total_pages = len(pdf)

    new_chunks = []


    for page_number, page in enumerate(
        pdf
    ):

        text = page.get_text(
            "text"
        )


        if not text.strip():

            continue


        page_chunks = chunk_text(
            text
        )


        for chunk in page_chunks:

            new_chunks.append(
                {
                    "text":
                        chunk,

                    "page":
                        page_number + 1,

                    "filename":
                        filename
                }
            )


    pdf.close()


    if len(new_chunks) == 0:

        raise ValueError(
            "No readable text found. "
            "The PDF may be scanned "
            "or image-only."
        )


    texts = [
        chunk["text"]
        for chunk in new_chunks
    ]


    embeddings = get_embeddings(
        texts
    )


    embeddings = np.array(
        embeddings,
        dtype="float32"
    )


    faiss.normalize_L2(
        embeddings
    )


    dimension = (
        embeddings.shape[1]
    )


    new_index = faiss.IndexFlatIP(
        dimension
    )


    new_index.add(
        embeddings
    )


    new_bm25 = build_bm25(
        new_chunks
    )


    with open(
        "pdf_chunks.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            new_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    faiss.write_index(
        new_index,
        "pdf_faiss.index"
    )


    return (
        new_chunks,
        new_index,
        new_bm25,
        total_pages
    )


# ============================================================
# 16. NORMALIZE SCORES
# ============================================================

def normalize_scores(
    scores
):

    if len(scores) == 0:

        return np.array([])


    scores = np.array(
        scores,
        dtype=float
    )


    minimum = scores.min()

    maximum = scores.max()


    if maximum == minimum:

        return np.zeros(
            len(scores)
        )


    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


# ============================================================
# 17. QUERY REWRITING
# ============================================================

def rewrite_query(
    question,
    history
):

    if len(history) == 0:

        return question


    history_text = ""


    for message in history[-6:]:

        history_text += (
            f"{message.role}: "
            f"{message.content}\n"
        )


    prompt = f"""
Rewrite the user's current question
as a standalone search query.

Use conversation history only to
resolve references such as:

"it"
"this"
"that"
"they"
"this concept"

Do not answer the question.

Conversation:

{history_text}

Current question:

{question}

Return only the standalone query.
"""


    try:

        response = client.responses.create(
            model=GENERATION_MODEL,
            input=prompt
        )


        rewritten = (
            response.output_text
            .strip()
        )


        if rewritten:

            return rewritten


    except Exception as error:

        print(
            "Query rewriting failed:",
            error
        )


    return question


# ============================================================
# 18. HYBRID SEARCH
# ============================================================

def hybrid_search(
    question
):

    if (
        index is None
        or
        bm25 is None
        or
        len(chunks) == 0
    ):

        return []


    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    question_embedding = get_embedding(
        question
    )


    question_embedding = np.array(
        [question_embedding],
        dtype="float32"
    )


    faiss.normalize_L2(
        question_embedding
    )


    vector_k = min(
        VECTOR_CANDIDATES,
        len(chunks)
    )


    vector_scores, vector_indices = (
        index.search(
            question_embedding,
            vector_k
        )
    )


    vector_results = {}


    for score, chunk_index in zip(
        vector_scores[0],
        vector_indices[0]
    ):

        if chunk_index == -1:

            continue


        vector_results[
            int(chunk_index)
        ] = float(score)


    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    query_tokens = (
        question
        .lower()
        .split()
    )


    keyword_scores = (
        bm25.get_scores(
            query_tokens
        )
    )


    keyword_k = min(
        BM25_CANDIDATES,
        len(chunks)
    )


    keyword_indices = np.argsort(
        keyword_scores
    )[::-1][:keyword_k]


    keyword_results = {}


    for chunk_index in (
        keyword_indices
    ):

        keyword_results[
            int(chunk_index)
        ] = float(
            keyword_scores[
                chunk_index
            ]
        )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    candidate_indices = list(
        set(
            vector_results.keys()
        )
        |
        set(
            keyword_results.keys()
        )
    )


    if not candidate_indices:

        return []


    raw_vector_scores = [

        vector_results.get(
            idx,
            0
        )

        for idx in candidate_indices
    ]


    raw_keyword_scores = [

        keyword_results.get(
            idx,
            0
        )

        for idx in candidate_indices
    ]


    normalized_vector = (
        normalize_scores(
            raw_vector_scores
        )
    )


    normalized_keyword = (
        normalize_scores(
            raw_keyword_scores
        )
    )


    results = []


    for position, chunk_index in enumerate(
        candidate_indices
    ):

        hybrid_score = (

            VECTOR_WEIGHT
            *
            normalized_vector[
                position
            ]

            +

            BM25_WEIGHT
            *
            normalized_keyword[
                position
            ]

        )


        results.append(
            (
                chunk_index,
                float(
                    hybrid_score
                )
            )
        )


    results.sort(
        key=lambda item:
            item[1],
        reverse=True
    )


    return results[
        :RETRIEVAL_K
    ]


# ============================================================
# 19. RERANKER
# ============================================================

def rerank_chunks(
    question,
    candidates
):

    if len(candidates) == 0:

        return []


    candidate_text = ""

    valid_ids = set()


    for (
        chunk_index,
        _
    ) in candidates:

        valid_ids.add(
            chunk_index
        )


        chunk_text = (
            chunks[
                chunk_index
            ]["text"]
        )


        candidate_text += f"""
Chunk ID: {chunk_index}

{chunk_text}

-------------------------
"""


    prompt = f"""
You are a document relevance reranker.

Evaluate every candidate based on how
useful it is for answering the question.

Question:

{question}

Candidate chunks:

{candidate_text}

Score every candidate from 0 to 10.

10 = directly answers the question
8 = highly relevant
6 = useful supporting information
4 = somewhat relevant
2 = weakly related
0 = irrelevant

Use the exact Chunk ID.
"""


    try:

        response = client.responses.parse(
            model=GENERATION_MODEL,
            input=prompt,
            text_format=RerankResult
        )


        parsed = response.output_parsed


        if parsed is None:

            return []


        results = []


        for item in parsed.results:

            if (
                item.chunk_index
                not in valid_ids
            ):

                continue


            results.append(
                (
                    item.chunk_index,
                    item.score
                )
            )


        results.sort(
            key=lambda item:
                item[1],
            reverse=True
        )


        return results


    except Exception as error:

        print(
            "Reranking failed:",
            error
        )


        return []


# ============================================================
# 20. MAIN RAG
# ============================================================

def ask_rag(
    question,
    history
):

    search_query = rewrite_query(
        question,
        history
    )


    candidates = hybrid_search(
        search_query
    )


    if len(candidates) == 0:

        return {
            "answer":
                "I don't know based "
                "on the provided PDF.",

            "source_pages":
                [],

            "search_query":
                search_query
        }


    reranked = rerank_chunks(
        search_query,
        candidates
    )


    if len(reranked) == 0:

        reranked = [

            (
                chunk_index,
                hybrid_score * 10
            )

            for (
                chunk_index,
                hybrid_score
            ) in candidates

        ]


    final_results = reranked[
        :FINAL_TOP_K
    ]


    relevant_chunks = []

    source_pages = []


    for (
        chunk_index,
        rerank_score
    ) in final_results:


        if (
            rerank_score
            <
            RERANK_THRESHOLD
        ):

            continue


        chunk = chunks[
            chunk_index
        ]


        text = chunk[
            "text"
        ]


        page = chunk[
            "page"
        ]


        relevant_chunks.append(
            f"Page {page}:\n{text}"
        )


        source_pages.append(
            page
        )


    if len(relevant_chunks) == 0:

        return {
            "answer":
                "I don't know based "
                "on the provided PDF.",

            "source_pages":
                [],

            "search_query":
                search_query
        }


    context = "\n\n".join(
        relevant_chunks
    )


    history_text = ""


    for message in history[-6:]:

        history_text += (
            f"{message.role}: "
            f"{message.content}\n"
        )


    prompt = f"""
You are an AI learning assistant.

Answer the user's question using ONLY
the PDF context.

Use conversation history only to
understand follow-up references.

Do not introduce facts unsupported
by the PDF.

Recent conversation:

{history_text}

PDF Context:

{context}

User Question:

{question}

Explain the answer clearly and simply.

If the PDF does not contain enough
information, say:

"I don't know based on the provided PDF."
"""


    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )


    return {
        "answer":
            response.output_text,

        "source_pages":
            sorted(
                set(
                    source_pages
                )
            ),

        "search_query":
            search_query
    }


# ============================================================
# 21. FRONTEND
# ============================================================

@app.get("/")
def frontend():

    return FileResponse(
        "frontend/index.html"
    )


# ============================================================
# 22. HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok",

        "document_loaded":
            index is not None,

        "chunks_loaded":
            len(chunks),

        "vectors_loaded":
            (
                index.ntotal
                if index is not None
                else 0
            )
    }


# ============================================================
# 23. UPLOAD
# ============================================================

@app.post(
    "/upload",
    response_model=
        UploadResponse
)
async def upload_pdf(
    file:
        UploadFile =
        File(...)
):

    global chunks
    global index
    global bm25


    filename = (
        file.filename
        or
        "document.pdf"
    )


    if not (
        filename
        .lower()
        .endswith(".pdf")
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files "
                "are supported."
            )
        )


    pdf_bytes = await file.read()


    if not pdf_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded PDF is empty."
            )
        )


    if (
        len(pdf_bytes)
        >
        MAX_UPLOAD_SIZE
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "PDF is too large."
            )
        )


    try:

        (
            new_chunks,
            new_index,
            new_bm25,
            page_count

        ) = build_pdf_knowledge_base(
            pdf_bytes,
            filename
        )


        chunks = new_chunks

        index = new_index

        bm25 = new_bm25


        return {
            "message":
                "PDF uploaded and "
                "indexed successfully.",

            "filename":
                filename,

            "pages":
                page_count,

            "chunks":
                len(chunks),

            "vectors":
                index.ntotal
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# 24. ASK
# ============================================================

@app.post(
    "/ask",
    response_model=
        QuestionResponse
)
def ask(
    request:
        QuestionRequest
):

    if (
        index is None
        or
        len(chunks) == 0
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No PDF loaded. "
                "Upload a PDF first."
            )
        )


    try:

        history = get_chat_history(
            request.session_id,
            limit=6
        )


        result = ask_rag(
            request.question,
            history
        )


        # Save user message
        save_message(
            request.session_id,
            "user",
            request.question,
            []
        )


        # Save AI message + citations
        save_message(
            request.session_id,
            "assistant",
            result["answer"],
            result["source_pages"]
        )


        return result


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# 25. GET HISTORY
# ============================================================

@app.get(
    "/history/{session_id}",
    response_model=
        HistoryResponse
)
def history(
    session_id: str
):

    messages = (
        get_all_chat_history(
            session_id
        )
    )


    return {
        "session_id":
            session_id,

        "messages":
            messages
    }


# ============================================================
# 26. DELETE HISTORY
# ============================================================

@app.delete(
    "/history/{session_id}"
)
def clear_history(
    session_id: str
):

    delete_chat_history(
        session_id
    )


    return {
        "message":
            "Conversation cleared."
    }