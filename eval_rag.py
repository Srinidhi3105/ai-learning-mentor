from openai import OpenAI
from dotenv import load_dotenv

import numpy as np
import faiss
import json


load_dotenv()

client = OpenAI()


# ----------------------------------------
# Generate embedding
# ----------------------------------------

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ----------------------------------------
# Load chunks
# ----------------------------------------

with open(
    "pdf_chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


# ----------------------------------------
# Load FAISS
# ----------------------------------------

index = faiss.read_index(
    "pdf_faiss.index"
)


# ----------------------------------------
# TEST DATA
# Change these based on YOUR PDF
# ----------------------------------------

test_cases = [

    {
        "question": "What is self-attention?",
        "expected_page": 12
    },

    {
        "question": "What is multi-head attention?",
        "expected_page": 13
    },

    {
        "question": "What are embeddings?",
        "expected_page": 20
    }

]


TOP_K = 3

passed = 0


# ----------------------------------------
# Run evaluation
# ----------------------------------------

for test in test_cases:

    question = test["question"]

    expected_page = test["expected_page"]


    # Create question embedding
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


    # Search
    scores, indices = index.search(
        embedding,
        TOP_K
    )


    retrieved_pages = []


    for chunk_index in indices[0]:

        if chunk_index == -1:
            continue

        page = chunks[
            chunk_index
        ]["page"]

        retrieved_pages.append(
            page
        )


    # ----------------------------------------
    # Check expected page
    # ----------------------------------------

    if expected_page in retrieved_pages:

        result = "PASS"

        passed += 1

    else:

        result = "FAIL"


    print("\nQuestion:")
    print(question)

    print(
        "Expected page:",
        expected_page
    )

    print(
        "Retrieved pages:",
        retrieved_pages
    )

    print(
        "Result:",
        result
    )


# ----------------------------------------
# Final score
# ----------------------------------------

total = len(
    test_cases
)


accuracy = (
    passed / total
) * 100


print("\n======================")
print("RAG RETRIEVAL RESULTS")
print("======================")

print(
    "Passed:",
    passed,
    "/",
    total
)

print(
    "Hit Rate:",
    round(
        accuracy,
        2
    ),
    "%"
)