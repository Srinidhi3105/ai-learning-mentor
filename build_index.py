"""
It loads a local document (notes.txt), breaks it down into small, overlapping chunks based on AI tokens, and converts those text chunks into mathematical vectors (embeddings) using the OpenAI API. Finally, it saves the vectors into an embeddings.npy file and the corresponding text into a chunks.json file on your hard drive, allowing you to search through your document later without having to pay to re-process it.
"""

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import tiktoken
import json

load_dotenv()

client = OpenAI()


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def chunk_text(text, chunk_size=200, overlap=40):

    encoding = tiktoken.get_encoding("o200k_base")

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(chunk_tokens)

        chunks.append(chunk)

        start = end - overlap

    return chunks


# Load file
text = load_file("notes.txt")


# Create chunks
chunks = chunk_text(
    text,
    chunk_size=200,
    overlap=40
)

print("Number of chunks:", len(chunks))


# Generate embeddings
embeddings = []

for i, chunk in enumerate(chunks):

    print(f"Embedding chunk {i + 1}")

    embedding = get_embedding(chunk)

    embeddings.append(embedding)


# Save embeddings
np.save(
    "embeddings.npy",
    np.array(embeddings)
)


# Save chunks
with open(
    "chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


print("\nIndex created successfully.")