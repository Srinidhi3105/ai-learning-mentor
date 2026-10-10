from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import numpy as np
import faiss
import json

from rank_bm25 import BM25Okapi


# ============================================================
# 1. SETUP
# ============================================================

load_dotenv()

client = OpenAI()


# ============================================================
# 2. STRUCTURED OUTPUT MODELS FOR RERANKING
# ============================================================

class RerankItem(BaseModel):
    chunk_index: int
    score: float = Field(ge=0, le=10)


class RerankResult(BaseModel):
    results: list[RerankItem]


# ============================================================
# 3. CREATE EMBEDDING
# ============================================================

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ============================================================
# 4. QUERY REWRITING
# ============================================================

def rewrite_query(question, history_text):

    # No history means no rewriting is needed
    if not history_text.strip():
        return question

    prompt = f"""
Rewrite the current user question as a standalone search query.

Use the conversation history only to understand references
such as "it", "this", "that", "they", or similar references.

Do not answer the question.

Conversation history:

{history_text}

Current question:

{question}

Return only the rewritten search query.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text.strip()


# ============================================================
# 5. NORMALIZE SCORES
# ============================================================

def normalize_scores(scores):

    scores = np.array(
        scores,
        dtype=float
    )

    minimum = scores.min()
    maximum = scores.max()

    # If every score is identical, the signal doesn't help ranking
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
# 6. STRUCTURED RERANKING
# ============================================================

def rerank_chunks(question, candidates, chunks):

    candidate_text = ""

    valid_candidate_ids = set()

    for candidate in candidates:

        chunk_index = candidate[0]

        valid_candidate_ids.add(
            chunk_index
        )

        chunk_text = chunks[
            chunk_index
        ]["text"]

        candidate_text += f"""
Chunk ID: {chunk_index}

{chunk_text}

-------------------------
"""


    prompt = f"""
You are a document relevance reranker.

Evaluate how useful each candidate chunk is for answering
the user's question.

Question:

{question}


Candidate chunks:

{candidate_text}


Give EVERY supplied chunk a relevance score from 0 to 10.

Scoring guide:

10 = directly answers the question
8 = extremely relevant
6 = useful supporting information
4 = somewhat related
2 = weakly related
0 = completely irrelevant

Use the exact Chunk ID supplied with each chunk.
"""


    try:

        response = client.responses.parse(
            model="gpt-4.1-mini",
            input=prompt,
            text_format=RerankResult
        )

        parsed = response.output_parsed


        if parsed is None:
            return []


        reranked = []


        for item in parsed.results:

            # Ignore any unexpected chunk IDs
            if item.chunk_index not in valid_candidate_ids:
                continue

            reranked.append(
                (
                    item.chunk_index,
                    item.score
                )
            )


        reranked.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return reranked


    except Exception as error:

        print(
            "\nReranker error:",
            error
        )

        return []


# ============================================================
# 7. LOAD PDF CHUNKS
# ============================================================

with open(
    "pdf_chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


# ============================================================
# 8. LOAD FAISS INDEX
# ============================================================

index = faiss.read_index(
    "pdf_faiss.index"
)


print(
    "Loaded PDF chunks:",
    len(chunks)
)

print(
    "Vectors in FAISS:",
    index.ntotal
)


# ============================================================
# 9. BUILD BM25 INDEX
# ============================================================

tokenized_chunks = []


for chunk in chunks:

    tokens = (
        chunk["text"]
        .lower()
        .split()
    )

    tokenized_chunks.append(
        tokens
    )


bm25 = BM25Okapi(
    tokenized_chunks
)


# ============================================================
# 10. SETTINGS
# ============================================================

VECTOR_CANDIDATES = 10

BM25_CANDIDATES = 10

RETRIEVAL_K = 10

FINAL_TOP_K = 3

VECTOR_WEIGHT = 0.7

BM25_WEIGHT = 0.3

RERANK_THRESHOLD = 4.0


# ============================================================
# 11. CONVERSATION MEMORY
# ============================================================

chat_history = []


print(
    "\nPDF Learning Assistant started."
)

print(
    "Type 'exit' to stop."
)


# ============================================================
# 12. MAIN CHAT LOOP
# ============================================================

while True:

    question = input(
        "\nYou: "
    )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if question.lower().strip() == "exit":

        print(
            "\nGoodbye!"
        )

        break


    # --------------------------------------------------------
    # BUILD RECENT CONVERSATION HISTORY
    # --------------------------------------------------------

    history_text = ""


    for message in chat_history[-6:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    # --------------------------------------------------------
    # QUERY REWRITING
    # --------------------------------------------------------

    rewritten_question = rewrite_query(
        question,
        history_text
    )


    print(
        "\nSearch query:",
        rewritten_question
    )


    # ========================================================
    # VECTOR / FAISS SEARCH
    # ========================================================

    question_embedding = get_embedding(
        rewritten_question
    )


    question_embedding = np.array(
        [question_embedding],
        dtype="float32"
    )


    faiss.normalize_L2(
        question_embedding
    )


    number_of_vector_results = min(
        VECTOR_CANDIDATES,
        len(chunks)
    )


    vector_scores, vector_indices = index.search(
        question_embedding,
        number_of_vector_results
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
        rewritten_question
        .lower()
        .split()
    )


    bm25_scores = bm25.get_scores(
        query_tokens
    )


    number_of_bm25_results = min(
        BM25_CANDIDATES,
        len(chunks)
    )


    bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:number_of_bm25_results]


    bm25_results = {}


    for idx in bm25_indices:

        bm25_results[
            int(idx)
        ] = float(
            bm25_scores[idx]
        )


    # ========================================================
    # COMBINE VECTOR + BM25 CANDIDATES
    # ========================================================

    candidate_indices = list(
        set(vector_results.keys())
        |
        set(bm25_results.keys())
    )


    vector_candidate_scores = [
        vector_results.get(
            idx,
            0
        )
        for idx in candidate_indices
    ]


    bm25_candidate_scores = [
        bm25_results.get(
            idx,
            0
        )
        for idx in candidate_indices
    ]


    normalized_vector_scores = normalize_scores(
        vector_candidate_scores
    )


    normalized_bm25_scores = normalize_scores(
        bm25_candidate_scores
    )


    hybrid_results = []


    for i, idx in enumerate(
        candidate_indices
    ):

        hybrid_score = (
            VECTOR_WEIGHT
            *
            normalized_vector_scores[i]
            +
            BM25_WEIGHT
            *
            normalized_bm25_scores[i]
        )


        hybrid_results.append(
            (
                idx,
                hybrid_score,
                vector_candidate_scores[i],
                bm25_candidate_scores[i]
            )
        )


    # --------------------------------------------------------
    # SORT HYBRID RESULTS
    # --------------------------------------------------------

    hybrid_results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    candidates = hybrid_results[
        :RETRIEVAL_K
    ]


    print(
        "\nHybrid retrieval candidates:"
    )


    for (
        chunk_index,
        hybrid_score,
        vector_score,
        bm25_score
    ) in candidates:

        print(
            "\nChunk:",
            chunk_index
        )

        print(
            "Hybrid:",
            round(
                hybrid_score,
                3
            )
        )

        print(
            "Vector:",
            round(
                vector_score,
                3
            )
        )

        print(
            "BM25:",
            round(
                bm25_score,
                3
            )
        )


    # ========================================================
    # STRUCTURED RERANKING
    # ========================================================

    reranked_results = rerank_chunks(
        rewritten_question,
        candidates,
        chunks
    )


    # Fallback if reranking fails
    if len(reranked_results) == 0:

        print(
            "\nReranking unavailable. "
            "Using hybrid ranking."
        )

        reranked_results = [
            (
                result[0],
                result[1] * 10
            )
            for result in candidates
        ]


    # --------------------------------------------------------
    # TAKE FINAL TOP-K
    # --------------------------------------------------------

    top_results = reranked_results[
        :FINAL_TOP_K
    ]


    relevant_chunks = []

    source_pages = []


    print(
        "\nFinal reranked results:\n"
    )


    for (
        chunk_index,
        rerank_score
    ) in top_results:


        chunk_data = chunks[
            chunk_index
        ]


        text = chunk_data[
            "text"
        ]

        page = chunk_data[
            "page"
        ]


        print(
            "Reranker score:",
            round(
                rerank_score,
                2
            )
        )

        print(
            "Page:",
            page
        )

        print(text)

        print(
            "-------------------------"
        )


        # Reject weak reranker results
        if rerank_score >= RERANK_THRESHOLD:

            relevant_chunks.append(
                f"Page {page}:\n{text}"
            )

            source_pages.append(
                page
            )


    # ========================================================
    # NO RELEVANT INFORMATION
    # ========================================================

    if len(relevant_chunks) == 0:

        answer = (
            "I don't know based on "
            "the provided PDF."
        )


        print(
            "\nAI:\n"
        )

        print(answer)


        chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        continue


    # ========================================================
    # BUILD FINAL CONTEXT
    # ========================================================

    context = "\n\n".join(
        relevant_chunks
    )


    # ========================================================
    # FINAL ANSWER PROMPT
    # ========================================================

    prompt = f"""
You are an AI learning assistant.

Answer the user's question using only the PDF context below.

Use the recent conversation only to understand follow-up
questions and references.

Do not introduce factual information that is unsupported
by the PDF context.


Recent Conversation:

{history_text}


PDF Context:

{context}


Current Question:

{question}


Explain the answer clearly and simply.

If the PDF does not contain enough information to answer
the question, say:

"I don't know based on the provided PDF."
"""


    # ========================================================
    # GENERATE FINAL ANSWER
    # ========================================================

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )


    answer = response.output_text


    # ========================================================
    # PRINT ANSWER
    # ========================================================

    print(
        "\nAI:\n"
    )

    print(answer)


    # ========================================================
    # PRINT SOURCE PAGES
    # ========================================================

    unique_pages = sorted(
        set(source_pages)
    )


    print(
        "\nSource pages:",
        ", ".join(
            str(page)
            for page in unique_pages
        )
    )


    # ========================================================
    # SAVE CONVERSATION MEMORY
    # ========================================================

    chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )