from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import numpy as np
import faiss
import json


load_dotenv()

client = OpenAI()


# ============================================================
# 1. STRUCTURED EVALUATION RESULT
# ============================================================

class EvaluationResult(BaseModel):
    correctness_score: float = Field(ge=0, le=10)
    groundedness_score: float = Field(ge=0, le=10)
    passed: bool
    reason: str


# ============================================================
# 2. EMBEDDING
# ============================================================

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ============================================================
# 3. LOAD PDF DATA
# ============================================================

with open(
    "pdf_chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


index = faiss.read_index(
    "pdf_faiss.index"
)


# ============================================================
# 4. RETRIEVAL FUNCTION
# ============================================================

def retrieve(question, top_k=3):

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

    scores, indices = index.search(
        embedding,
        top_k
    )

    retrieved_chunks = []

    retrieved_pages = []


    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        if chunk_index == -1:
            continue

        chunk = chunks[
            chunk_index
        ]

        retrieved_chunks.append(
            chunk["text"]
        )

        retrieved_pages.append(
            chunk["page"]
        )


    return (
        retrieved_chunks,
        retrieved_pages
    )


# ============================================================
# 5. GENERATE RAG ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are an AI learning assistant.

Answer the question using only the context below.

Context:

{context}

Question:

{question}

If the answer is not supported by the context,
say:

"I don't know based on the provided PDF."
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return (
        response.output_text,
        context
    )


# ============================================================
# 6. LLM JUDGE
# ============================================================

def evaluate_answer(
    question,
    context,
    generated_answer,
    expected_answer
):

    prompt = f"""
Evaluate this RAG response.

Question:

{question}

Retrieved Context:

{context}

Expected Answer:

{expected_answer}

Generated Answer:

{generated_answer}


Evaluate:

1. Correctness
How accurately does the generated answer answer
the user's question?

2. Groundedness
Is every factual claim supported by the
retrieved context?

Give both scores from 0 to 10.

passed should only be true when:

correctness_score >= 7

AND

groundedness_score >= 7
"""


    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=prompt,
        text_format=EvaluationResult
    )

    return response.output_parsed


# ============================================================
# 7. TEST DATA
# ============================================================

# IMPORTANT:
#
# Change these questions, pages and answers
# so they match YOUR PDF.

test_cases = [

    {
        "question":
            "What is self-attention?",

        "expected_page":
            12,

        "expected_answer":
            "Self-attention allows tokens to "
            "consider other tokens in the sequence."
    },

    {
        "question":
            "What is multi-head attention?",

        "expected_page":
            13,

        "expected_answer":
            "Multi-head attention uses multiple "
            "attention heads in parallel."
    },

    {
        "question":
            "What are embeddings?",

        "expected_page":
            20,

        "expected_answer":
            "Embeddings represent information "
            "as numerical vectors."
    }

]


# ============================================================
# 8. METRICS
# ============================================================

retrieval_passes = 0

answer_passes = 0

correctness_scores = []

groundedness_scores = []


# ============================================================
# 9. RUN TEST SUITE
# ============================================================

for number, test in enumerate(
    test_cases,
    start=1
):

    print(
        "\n======================================"
    )

    print(
        f"TEST {number}"
    )

    print(
        "======================================"
    )


    question = test[
        "question"
    ]

    expected_page = test[
        "expected_page"
    ]

    expected_answer = test[
        "expected_answer"
    ]


    print(
        "\nQuestion:",
        question
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved_chunks, retrieved_pages = retrieve(
        question,
        top_k=3
    )


    retrieval_passed = (
        expected_page in retrieved_pages
    )


    if retrieval_passed:
        retrieval_passes += 1


    print(
        "\nExpected page:",
        expected_page
    )

    print(
        "Retrieved pages:",
        retrieved_pages
    )

    print(
        "Retrieval:",
        "PASS" if retrieval_passed else "FAIL"
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    generated_answer, context = generate_answer(
        question,
        retrieved_chunks
    )


    print(
        "\nGenerated answer:"
    )

    print(
        generated_answer
    )


    # --------------------------------------------------------
    # Judge answer
    # --------------------------------------------------------

    evaluation = evaluate_answer(
        question,
        context,
        generated_answer,
        expected_answer
    )


    correctness_scores.append(
        evaluation.correctness_score
    )

    groundedness_scores.append(
        evaluation.groundedness_score
    )


    if evaluation.passed:
        answer_passes += 1


    print(
        "\nCorrectness:",
        evaluation.correctness_score
    )

    print(
        "Groundedness:",
        evaluation.groundedness_score
    )

    print(
        "Answer:",
        "PASS" if evaluation.passed else "FAIL"
    )

    print(
        "Reason:",
        evaluation.reason
    )


# ============================================================
# 10. FINAL REPORT
# ============================================================

total_tests = len(
    test_cases
)


retrieval_hit_rate = (
    retrieval_passes
    /
    total_tests
) * 100


answer_pass_rate = (
    answer_passes
    /
    total_tests
) * 100


average_correctness = (
    sum(correctness_scores)
    /
    len(correctness_scores)
)


average_groundedness = (
    sum(groundedness_scores)
    /
    len(groundedness_scores)
)


print(
    "\n\n======================================"
)

print(
    "FINAL RAG EVALUATION REPORT"
)

print(
    "======================================"
)


print(
    "\nRetrieval Hit@3:",
    round(
        retrieval_hit_rate,
        2
    ),
    "%"
)


print(
    "Average Correctness:",
    round(
        average_correctness,
        2
    ),
    "/ 10"
)


print(
    "Average Groundedness:",
    round(
        average_groundedness,
        2
    ),
    "/ 10"
)


print(
    "Answer Pass Rate:",
    round(
        answer_pass_rate,
        2
    ),
    "%"
)