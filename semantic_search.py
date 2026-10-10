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
    "Python is a popular programming language.",
    "Dogs are loyal domestic animals.",
    "The stock market can be volatile.",
    "Transformers are widely used in large language models."
]


# Create embeddings for our documents
document_embeddings = []

for document in documents:
    embedding = get_embedding(document)
    document_embeddings.append(embedding)


# Ask the user a question
query = input("Ask a question: ")

query_embedding = get_embedding(query)


# Find the most similar document
best_score = -1
best_document = None

for document, embedding in zip(documents, document_embeddings):

    score = cosine_similarity(
        query_embedding,
        embedding
    )

    print(document, "->", score)

    if score > best_score:
        best_score = score
        best_document = document


print("\nBest matching document:")
print(best_document)

print("\nSimilarity score:")
print(best_score)