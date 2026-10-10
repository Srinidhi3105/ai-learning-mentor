from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


documents = [
    "Bengaluru is the capital of Karnataka.",

    "Python is a popular programming language used in AI and data science.",

    "Transformers use self-attention to understand relationships between tokens.",

    "Multi-head attention allows transformers to learn different relationships between tokens.",

    "Large language models use transformer architectures to process and generate text.",

    "Dogs are loyal domestic animals.",

    "The stock market can be volatile."
]


# ----------------------------------------
# STEP 1: Create embeddings for documents
# ----------------------------------------

document_embeddings = []

for doc in documents:
    embedding = get_embedding(doc)
    document_embeddings.append(embedding)


# ----------------------------------------
# STEP 2: Get user question
# ----------------------------------------

question = input("Ask your question: ")

question_embedding = get_embedding(question)


# ----------------------------------------
# STEP 3: Compare question with documents
# ----------------------------------------

results = []

for doc, emb in zip(documents, document_embeddings):

    score = cosine_similarity(
        question_embedding,
        emb
    )

    results.append((doc, score))


# ----------------------------------------
# STEP 4: Sort by similarity
# ----------------------------------------

results.sort(
    key=lambda x: x[1],
    reverse=True
)


# ----------------------------------------
# STEP 5: Get Top-K documents
# ----------------------------------------

TOP_K = 3
SIMILARITY_THRESHOLD = 0.35

top_results = results[:TOP_K]

relevant_documents = []


print("\nTop retrieved documents:\n")

for doc, score in top_results:

    print("Document:", doc)
    print("Score:", score)
    print()

    if score >= SIMILARITY_THRESHOLD:
        relevant_documents.append(doc)


# ----------------------------------------
# STEP 6: Check if anything relevant exists
# ----------------------------------------

if len(relevant_documents) == 0:

    print(
        "\nAI Answer: "
        "I don't know based on the provided documents."
    )

else:

    # ----------------------------------------
    # STEP 7: Combine retrieved documents
    # ----------------------------------------

    context = "\n\n".join(relevant_documents)


    # ----------------------------------------
    # STEP 8: Build prompt
    # ----------------------------------------

    prompt = f"""
Answer the user's question using only the context below.

Context:

{context}

Question:

{question}

If the answer is not present in the context,
say: "I don't know based on the provided context."
"""


    # ----------------------------------------
    # STEP 9: Send context to LLM
    # ----------------------------------------

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )


    # ----------------------------------------
    # STEP 10: Print answer
    # ----------------------------------------

    print("\nAI Answer:")
    print(response.output_text)