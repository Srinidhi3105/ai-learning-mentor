"""
• Loads Pre-built Data: It reads the pre-saved text segments from chunks.json and their matching mathematical vectors from embeddings.npy.
• Vectorizes Your Question: When you type a question, it calls the OpenAI API just once to convert your query into a single vector embedding.
• Calculates Similarity: It compares your question's vector against all the document vectors using cosine_similarity to see which pieces of text closely match your intent.
• Filters and Ranks: It sorts the document chunks from best match to worst match, grabs the top 3 results, and filters out any chunk with a match score lower than 0.30.
• Generates the Answer: If relevant information is found, it copies those exact text chunks into a custom prompt and feeds them to an OpenAI model (gpt-4.1-mini) to write a simple, accurate response restricted only to that text. If no matches pass the threshold, it immediately stops and safely states it doesn't know the answer.
"""

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import faiss

load_dotenv()

client = OpenAI()


# ----------------------------------------
# 1. Generate embedding
# ----------------------------------------

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ----------------------------------------
# 2. Load chunks
# ----------------------------------------

with open(
    "chunks.json",
    "r",
    encoding="utf-8"
) as file:

    chunks = json.load(file)


# ----------------------------------------
# 3. Load FAISS index
# ----------------------------------------

index = faiss.read_index(
    "faiss.index"
)

print("Loaded chunks:", len(chunks))
print("Vectors in FAISS:", index.ntotal)


# ----------------------------------------
# 4. Ask user question
# ----------------------------------------

question = input(
    "\nAsk your question: "
)


# ----------------------------------------
# 5. Generate question embedding
# ----------------------------------------

question_embedding = get_embedding(
    question
)

question_embedding = np.array(
    [question_embedding],
    dtype="float32"
)


# ----------------------------------------
# 6. Normalize question vector
# ----------------------------------------

faiss.normalize_L2(
    question_embedding
)


# ----------------------------------------
# 7. Search FAISS
# ----------------------------------------

TOP_K = 3

scores, indices = index.search(
    question_embedding,
    TOP_K
)


# ----------------------------------------
# 8. Retrieve chunks
# ----------------------------------------

SIMILARITY_THRESHOLD = 0.30

relevant_chunks = []


print(
    "\nRetrieved chunks:\n"
)


for score, chunk_index in zip(
    scores[0],
    indices[0]
):

    chunk = chunks[chunk_index]

    print(
        "Score:",
        score
    )

    print(chunk)

    print(
        "-------------------------"
    )

    if score >= SIMILARITY_THRESHOLD:

        relevant_chunks.append(
            chunk
        )


# ----------------------------------------
# 9. Check if anything relevant exists
# ----------------------------------------

if len(relevant_chunks) == 0:

    print(
        "\nAI Answer: "
        "I don't know based on the provided document."
    )


else:

    # ----------------------------------------
    # 10. Combine context
    # ----------------------------------------

    context = "\n\n".join(
        relevant_chunks
    )


    # ----------------------------------------
    # 11. Create prompt
    # ----------------------------------------

    prompt = f"""
You are an AI learning assistant.

Answer the user's question using only the context below.

Context:

{context}

Question:

{question}

Explain the answer clearly and simply.

If the answer is not available in the context,
say:

"I don't know based on the provided document."
"""


    # ----------------------------------------
    # 12. Ask LLM
    # ----------------------------------------

    response = client.responses.create(

        model="gpt-4.1-mini",

        input=prompt

    )


    # ----------------------------------------
    # 13. Print answer
    # ----------------------------------------

    print(
        "\nAI Answer:\n"
    )

    print(
        response.output_text
    )