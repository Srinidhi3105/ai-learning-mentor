import numpy as np
import faiss

from openai import OpenAI
from dotenv import load_dotenv


from backend.config import (
    GENERATION_MODEL,
    VECTOR_CANDIDATES,
    BM25_CANDIDATES,
    RETRIEVAL_K,
    FINAL_TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RERANK_THRESHOLD
)

from backend.embeddings import (
    get_embedding
)

from backend.models import (
    RerankResult
)

from backend.observability import (
    logger,
    start_timer,
    elapsed_ms
)


load_dotenv()

client = OpenAI()


# ============================================================
# 1. NORMALIZE SCORES
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
# 2. QUERY REWRITING
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
Rewrite the current question as a standalone
search query.

Use conversation history only to resolve
references such as:

"it"
"this"
"that"
"they"
"this concept"

Do not answer the question.

Conversation:

{history_text}

Question:

{question}

Return only the rewritten search query.
"""


    try:

        start = start_timer()


        response = client.responses.create(
            model=GENERATION_MODEL,
            input=prompt
        )


        rewritten = (
            response.output_text
            .strip()
        )


        logger.info(
            f"QUERY_REWRITE_COMPLETE "
            f"duration_ms={elapsed_ms(start)}"
        )


        if rewritten:

            return rewritten


    except Exception as error:

        logger.exception(
            f"QUERY_REWRITE_FAILED "
            f"error={error}"
        )


    return question


# ============================================================
# 3. HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    chunks,
    index,
    bm25
):

    search_start = start_timer()


    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    embedding = get_embedding(
        question
    )


    embedding = np.array(
        [embedding],
        dtype="float32"
    )


    faiss.normalize_L2(
        embedding
    )


    vector_k = min(
        VECTOR_CANDIDATES,
        len(chunks)
    )


    vector_scores, vector_indices = (
        index.search(
            embedding,
            vector_k
        )
    )


    vector_results = {}


    for score, idx in zip(
        vector_scores[0],
        vector_indices[0]
    ):

        if idx == -1:

            continue


        vector_results[
            int(idx)
        ] = float(score)


    # ========================================================
    # BM25 SEARCH
    # ========================================================

    query_tokens = (
        question
        .lower()
        .split()
    )


    bm25_scores = (
        bm25.get_scores(
            query_tokens
        )
    )


    bm25_k = min(
        BM25_CANDIDATES,
        len(chunks)
    )


    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:bm25_k]


    bm25_results = {}


    for idx in bm25_indices:

        bm25_results[
            int(idx)
        ] = float(
            bm25_scores[idx]
        )


    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    candidate_indices = list(

        set(
            vector_results.keys()
        )

        |

        set(
            bm25_results.keys()
        )

    )


    if not candidate_indices:

        logger.info(
            "HYBRID_SEARCH_NO_RESULTS"
        )

        return []


    vector_values = [

        vector_results.get(
            idx,
            0
        )

        for idx in candidate_indices
    ]


    bm25_values = [

        bm25_results.get(
            idx,
            0
        )

        for idx in candidate_indices
    ]


    vector_normalized = (
        normalize_scores(
            vector_values
        )
    )


    bm25_normalized = (
        normalize_scores(
            bm25_values
        )
    )


    results = []


    for i, chunk_index in enumerate(
        candidate_indices
    ):

        hybrid_score = (

            VECTOR_WEIGHT
            *
            vector_normalized[i]

            +

            BM25_WEIGHT
            *
            bm25_normalized[i]

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


    final_results = results[
        :RETRIEVAL_K
    ]


    logger.info(
        f"HYBRID_SEARCH_COMPLETE "
        f"vector_candidates="
        f"{len(vector_results)} "
        f"bm25_candidates="
        f"{len(bm25_results)} "
        f"combined="
        f"{len(candidate_indices)} "
        f"returned="
        f"{len(final_results)} "
        f"duration_ms="
        f"{elapsed_ms(search_start)}"
    )


    return final_results


# ============================================================
# 4. RERANKING
# ============================================================

def rerank_chunks(
    question,
    candidates,
    chunks
):

    if not candidates:

        return []


    rerank_start = start_timer()


    candidate_text = ""

    valid_ids = set()


    for (
        chunk_index,
        _
    ) in candidates:

        valid_ids.add(
            chunk_index
        )


        text = chunks[
            chunk_index
        ]["text"]


        candidate_text += f"""
Chunk ID: {chunk_index}

{text}

-------------------------
"""


    prompt = f"""
You are a document relevance reranker.

Question:

{question}

Candidate chunks:

{candidate_text}

Score EVERY candidate from 0 to 10.

10 = directly answers the question
8 = highly relevant
6 = useful supporting information
4 = somewhat related
2 = weakly related
0 = irrelevant

Use the exact Chunk ID provided.
"""


    try:

        response = client.responses.parse(
            model=GENERATION_MODEL,
            input=prompt,
            text_format=RerankResult
        )


        parsed = (
            response.output_parsed
        )


        if parsed is None:

            logger.warning(
                "RERANK_NO_PARSED_OUTPUT"
            )

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


        logger.info(
            f"RERANK_COMPLETE "
            f"input_candidates="
            f"{len(candidates)} "
            f"output_results="
            f"{len(results)} "
            f"duration_ms="
            f"{elapsed_ms(rerank_start)}"
        )


        return results


    except Exception as error:

        logger.exception(
            f"RERANK_FAILED "
            f"duration_ms="
            f"{elapsed_ms(rerank_start)} "
            f"error={error}"
        )


        return []


# ============================================================
# 5. MAIN RAG PIPELINE
# ============================================================

def ask_rag(
    question,
    history,
    chunks,
    index,
    bm25
):

    rag_start = start_timer()


    logger.info(
        f"RAG_START "
        f"question={question!r}"
    )


    # ========================================================
    # QUERY REWRITING
    # ========================================================

    search_query = rewrite_query(
        question,
        history
    )


    logger.info(
        f"QUERY_REWRITE "
        f"original={question!r} "
        f"rewritten={search_query!r}"
    )


    # ========================================================
    # HYBRID RETRIEVAL
    # ========================================================

    candidates = hybrid_search(
        search_query,
        chunks,
        index,
        bm25
    )


    logger.info(
        f"HYBRID_SEARCH "
        f"candidates={len(candidates)}"
    )


    if not candidates:

        answer = (
            "I don't know based on "
            "the provided PDF."
        )


        logger.info(
            f"RAG_COMPLETE "
            f"result=no_candidates "
            f"duration_ms="
            f"{elapsed_ms(rag_start)}"
        )


        return {

            "answer":
                answer,

            "source_pages":
                [],

            "search_query":
                search_query

        }


    # ========================================================
    # RERANK
    # ========================================================

    reranked = rerank_chunks(
        search_query,
        candidates,
        chunks
    )


    logger.info(
        f"RERANK "
        f"results={len(reranked)}"
    )


    # ========================================================
    # FALLBACK TO HYBRID SCORES
    # ========================================================

    if not reranked:

        logger.warning(
            "Using hybrid ranking "
            "because reranker returned "
            "no results."
        )


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


    # ========================================================
    # FINAL TOP K
    # ========================================================

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


    source_pages = sorted(
        set(source_pages)
    )


    logger.info(
        f"RETRIEVED_PAGES "
        f"pages={source_pages}"
    )


    # ========================================================
    # NO RELEVANT CHUNKS
    # ========================================================

    if not relevant_chunks:

        answer = (
            "I don't know based on "
            "the provided PDF."
        )


        logger.info(
            f"RAG_COMPLETE "
            f"result=no_relevant_chunks "
            f"duration_ms="
            f"{elapsed_ms(rag_start)}"
        )


        return {

            "answer":
                answer,

            "source_pages":
                [],

            "search_query":
                search_query

        }


    # ========================================================
    # BUILD FINAL CONTEXT
    # ========================================================

    context = "\n\n".join(
        relevant_chunks
    )


    history_text = ""


    for message in history[-6:]:

        history_text += (
            f"{message.role}: "
            f"{message.content}\n"
        )


    # ========================================================
    # FINAL GENERATION
    # ========================================================

    prompt = f"""
You are an AI learning assistant.

Answer using ONLY the PDF context below.

Conversation history may only be used
to understand follow-up references.

Do not introduce unsupported factual
information.

Recent conversation:

{history_text}

PDF Context:

{context}

Question:

{question}

Explain the answer clearly and simply.

If the context is insufficient, say:

"I don't know based on the provided PDF."
"""


    generation_start = start_timer()


    response = client.responses.create(
        model=GENERATION_MODEL,
        input=prompt
    )


    answer = (
        response.output_text
    )


    logger.info(
        f"GENERATION_COMPLETE "
        f"duration_ms="
        f"{elapsed_ms(generation_start)}"
    )


    logger.info(
        f"RAG_COMPLETE "
        f"duration_ms="
        f"{elapsed_ms(rag_start)} "
        f"sources={source_pages}"
    )


    return {

        "answer":
            answer,

        "source_pages":
            source_pages,

        "search_query":
            search_query

    }