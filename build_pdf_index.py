from openai import OpenAI
from dotenv import load_dotenv
import pymupdf
import tiktoken
import numpy as np
import faiss
import json


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

client = OpenAI()


# -----------------------------------
# 2. Configuration
# -----------------------------------

PDF_FILE = "document.pdf"

CHUNK_SIZE = 200

OVERLAP = 40


# -----------------------------------
# 3. Generate embedding
# -----------------------------------

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# -----------------------------------
# 4. Token-based chunking
# -----------------------------------

def chunk_text(text, chunk_size=200, overlap=40):

    encoding = tiktoken.get_encoding("o200k_base")

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk = encoding.decode(chunk_tokens).strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# -----------------------------------
# 5. Open PDF
# -----------------------------------

pdf = pymupdf.open(PDF_FILE)

print("PDF pages:", len(pdf))

all_chunks = []


# -----------------------------------
# 6. Extract text page-by-page
# -----------------------------------

for page_number, page in enumerate(pdf):

    text = page.get_text("text")

    # Skip pages with no extracted text
    if not text.strip():
        continue

    page_chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )

    for chunk in page_chunks:

        all_chunks.append(
            {
                "text": chunk,
                "page": page_number + 1
            }
        )


pdf.close()


print("Total chunks:", len(all_chunks))


# -----------------------------------
# 7. Safety check
# -----------------------------------

if len(all_chunks) == 0:

    print(
        "No text could be extracted from the PDF."
    )

    exit()


# -----------------------------------
# 8. Create embeddings for chunks
# -----------------------------------

embeddings = []


for i, chunk_data in enumerate(all_chunks):

    print(
        f"Embedding chunk "
        f"{i + 1}/{len(all_chunks)} "
        f"- page {chunk_data['page']}"
    )

    embedding = get_embedding(
        chunk_data["text"]
    )

    embeddings.append(
        embedding
    )


# -----------------------------------
# 9. Convert embeddings to NumPy
# -----------------------------------

embeddings = np.array(
    embeddings,
    dtype="float32"
)


print(
    "Embedding shape:",
    embeddings.shape
)


# -----------------------------------
# 10. Normalize vectors
# -----------------------------------

faiss.normalize_L2(
    embeddings
)


# -----------------------------------
# 11. Create FAISS index
# -----------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)


# -----------------------------------
# 12. Add vectors to FAISS
# -----------------------------------

index.add(
    embeddings
)


print(
    "Vectors stored in FAISS:",
    index.ntotal
)


# -----------------------------------
# 13. Save FAISS index
# -----------------------------------

faiss.write_index(
    index,
    "pdf_faiss.index"
)


# -----------------------------------
# 14. Save chunks + metadata
# -----------------------------------

with open(
    "pdf_chunks.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=2
    )


# -----------------------------------
# 15. Finished
# -----------------------------------

print(
    "\nPDF index created successfully."
)

print(
    "Created: pdf_faiss.index"
)

print(
    "Created: pdf_chunks.json"
)