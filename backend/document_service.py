import json

import faiss
import numpy as np
import pymupdf
import tiktoken

from rank_bm25 import BM25Okapi

from backend.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKS_FILE,
    FAISS_FILE
)

from backend.embeddings import get_embeddings


def chunk_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    encoding = tiktoken.get_encoding(
        "o200k_base"
    )

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(
            chunk_tokens
        ).strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


def build_bm25(chunks):

    tokenized_chunks = []

    for chunk in chunks:

        tokens = (
            chunk["text"]
            .lower()
            .split()
        )

        tokenized_chunks.append(tokens)

    return BM25Okapi(
        tokenized_chunks
    )


def build_pdf_knowledge_base(
    pdf_bytes,
    filename
):

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf)

    chunks = []


    # --------------------------------
    # Extract pages
    # --------------------------------

    for page_number, page in enumerate(pdf):

        text = page.get_text("text")

        if not text.strip():
            continue

        page_chunks = chunk_text(text)

        for chunk in page_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "page": page_number + 1,
                    "filename": filename
                }
            )


    pdf.close()


    if len(chunks) == 0:

        raise ValueError(
            "No readable text was found in the PDF."
        )


    # --------------------------------
    # Generate embeddings
    # --------------------------------

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = get_embeddings(texts)

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )


    # --------------------------------
    # Normalize embeddings
    # --------------------------------

    faiss.normalize_L2(
        embeddings
    )


    # --------------------------------
    # Create FAISS index
    # --------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )


    # --------------------------------
    # BM25
    # --------------------------------

    bm25 = build_bm25(
        chunks
    )


    # --------------------------------
    # Save chunks
    # --------------------------------

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )


    # --------------------------------
    # Save FAISS
    # --------------------------------

    faiss.write_index(
        index,
        FAISS_FILE
    )


    return (
        chunks,
        index,
        bm25,
        total_pages
    )