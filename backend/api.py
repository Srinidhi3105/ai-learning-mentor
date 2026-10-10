from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from fastapi.responses import (
    FileResponse
)

import json
import os
import faiss


from backend.config import (
    MAX_UPLOAD_SIZE,
    CHUNKS_FILE,
    FAISS_FILE
)

from backend.models import (
    QuestionRequest,
    QuestionResponse,
    UploadResponse,
    HistoryResponse
)

from backend.database import (
    init_database,
    save_message,
    get_chat_history,
    get_all_chat_history,
    delete_chat_history
)

from backend.document_service import (
    build_pdf_knowledge_base,
    build_bm25
)

from backend.rag_service import (
    ask_rag
)

from backend.observability import (
    logger,
    start_timer,
    elapsed_ms
)


# ============================================================
# 1. FASTAPI
# ============================================================

app = FastAPI(
    title="AI Learning Mentor",
    version="1.0"
)


# ============================================================
# 2. REQUEST LOGGING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def log_requests(
    request: Request,
    call_next
):

    start = start_timer()


    logger.info(
        f"REQUEST "
        f"method={request.method} "
        f"path={request.url.path}"
    )


    try:

        response = await call_next(
            request
        )


        duration = elapsed_ms(
            start
        )


        logger.info(
            f"RESPONSE "
            f"method={request.method} "
            f"path={request.url.path} "
            f"status={response.status_code} "
            f"duration_ms={duration}"
        )


        return response


    except Exception as error:

        duration = elapsed_ms(
            start
        )


        logger.exception(
            f"REQUEST_FAILED "
            f"method={request.method} "
            f"path={request.url.path} "
            f"duration_ms={duration} "
            f"error={error}"
        )


        raise


# ============================================================
# 3. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8000"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# 4. INITIALIZE DATABASE
# ============================================================

init_database()


# ============================================================
# 5. CURRENT KNOWLEDGE BASE
# ============================================================

chunks = []

index = None

bm25 = None


# ============================================================
# 6. LOAD EXISTING KNOWLEDGE BASE
# ============================================================

def load_existing_knowledge_base():

    global chunks
    global index
    global bm25


    if not os.path.exists(
        CHUNKS_FILE
    ):

        logger.info(
            "No saved chunks found."
        )

        return


    if not os.path.exists(
        FAISS_FILE
    ):

        logger.info(
            "No saved FAISS index found."
        )

        return


    try:

        with open(
            CHUNKS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            loaded_chunks = (
                json.load(file)
            )


        loaded_index = (
            faiss.read_index(
                FAISS_FILE
            )
        )


        if (
            len(loaded_chunks)
            != loaded_index.ntotal
        ):

            logger.warning(
                "Chunk count does not "
                "match FAISS vector count."
            )

            return


        loaded_bm25 = build_bm25(
            loaded_chunks
        )


        chunks = loaded_chunks

        index = loaded_index

        bm25 = loaded_bm25


        logger.info(
            f"KNOWLEDGE_BASE_LOADED "
            f"chunks={len(chunks)} "
            f"vectors={index.ntotal}"
        )


    except Exception as error:

        logger.exception(
            f"KNOWLEDGE_BASE_LOAD_FAILED "
            f"error={error}"
        )


load_existing_knowledge_base()


# ============================================================
# 7. FRONTEND
# ============================================================

@app.get("/")
def frontend():

    return FileResponse(
        "frontend/index.html"
    )


# ============================================================
# 8. HEALTH
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
# 9. UPLOAD PDF
# ============================================================

@app.post(
    "/upload",
    response_model=UploadResponse
)
async def upload_pdf(

    file:
        UploadFile =
        File(...)

):

    global chunks
    global index
    global bm25


    upload_start = start_timer()


    filename = (
        file.filename
        or
        "document.pdf"
    )


    logger.info(
        f"PDF_UPLOAD_START "
        f"filename={filename!r}"
    )


    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    if not (
        filename
        .lower()
        .endswith(".pdf")
    ):

        logger.warning(
            f"PDF_UPLOAD_REJECTED "
            f"filename={filename!r} "
            f"reason=invalid_file_type"
        )


        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF files "
                "are supported."
            )
        )


    # --------------------------------------------------------
    # Read PDF
    # --------------------------------------------------------

    pdf_bytes = (
        await file.read()
    )


    # --------------------------------------------------------
    # Empty check
    # --------------------------------------------------------

    if not pdf_bytes:

        logger.warning(
            f"PDF_UPLOAD_REJECTED "
            f"filename={filename!r} "
            f"reason=empty_file"
        )


        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded PDF is empty."
            )
        )


    # --------------------------------------------------------
    # Size check
    # --------------------------------------------------------

    if (
        len(pdf_bytes)
        >
        MAX_UPLOAD_SIZE
    ):

        logger.warning(
            f"PDF_UPLOAD_REJECTED "
            f"filename={filename!r} "
            f"reason=file_too_large"
        )


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


        logger.info(
            f"PDF_UPLOAD_COMPLETE "
            f"filename={filename!r} "
            f"pages={page_count} "
            f"chunks={len(chunks)} "
            f"vectors={index.ntotal} "
            f"duration_ms="
            f"{elapsed_ms(upload_start)}"
        )


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

        logger.exception(
            f"PDF_UPLOAD_FAILED "
            f"filename={filename!r} "
            f"duration_ms="
            f"{elapsed_ms(upload_start)} "
            f"error={error}"
        )


        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# 10. ASK QUESTION
# ============================================================

@app.post(
    "/ask",
    response_model=QuestionResponse
)
def ask(

    request:
        QuestionRequest

):

    if (
        index is None
        or
        bm25 is None
        or
        len(chunks) == 0
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "No PDF is loaded. "
                "Upload a PDF first."
            )
        )


    try:

        logger.info(
            f"ASK_START "
            f"session_id={request.session_id!r}"
        )


        # ----------------------------------------------------
        # Load conversation memory
        # ----------------------------------------------------

        history = get_chat_history(
            request.session_id,
            limit=6
        )


        logger.info(
            f"HISTORY_LOADED "
            f"messages={len(history)}"
        )


        # ----------------------------------------------------
        # Run RAG
        # ----------------------------------------------------

        result = ask_rag(

            request.question,

            history,

            chunks,

            index,

            bm25

        )


        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        save_message(
            request.session_id,
            "user",
            request.question,
            []
        )


        # ----------------------------------------------------
        # Save assistant answer
        # ----------------------------------------------------

        save_message(
            request.session_id,
            "assistant",
            result["answer"],
            result["source_pages"]
        )


        logger.info(
            f"ASK_COMPLETE "
            f"session_id="
            f"{request.session_id!r} "
            f"sources="
            f"{result['source_pages']}"
        )


        return result


    except Exception as error:

        logger.exception(
            f"ASK_FAILED "
            f"session_id="
            f"{request.session_id!r} "
            f"error={error}"
        )


        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# 11. GET CHAT HISTORY
# ============================================================

@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse
)
def history(
    session_id: str
):

    messages = (
        get_all_chat_history(
            session_id
        )
    )


    logger.info(
        f"HISTORY_REQUEST "
        f"session_id={session_id!r} "
        f"messages={len(messages)}"
    )


    return {

        "session_id":
            session_id,

        "messages":
            messages
    }


# ============================================================
# 12. DELETE CHAT HISTORY
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


    logger.info(
        f"HISTORY_CLEARED "
        f"session_id={session_id!r}"
    )


    return {

        "message":
            "Conversation cleared."

    }