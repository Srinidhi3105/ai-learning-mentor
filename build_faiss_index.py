"""  
Script introduces FAISS (Facebook AI Similarity Search), replacing the manual, slow Python loops you used previously to search for matching text. FAISS allows you to perform lightning-fast vector calculations [1] on thousands or millions of documents in seconds, making it ideal for real-time applications. This script demonstrates how to build a FAISS index from pre-computed embeddings and save it for later use.
This script creares a high performance vector database using Facebook's Faiss library to enable lightning fast text searches.
It loads your pre-saved vectors from embeddings.npy and they are converted to float.
t then builds a specialized FAISS Inner Product index, injects all your document vectors into it, and saves this index directly to your hard drive as faiss.index. This eliminates the need for slow Python math loops when searching through large amounts of data.
"""

import faiss
import numpy as np


# ----------------------------------------
# 1. Load existing embeddings
# ----------------------------------------

embeddings = np.load("embeddings.npy")

print("Embeddings shape:")
print(embeddings.shape)


# FAISS expects float32
embeddings = embeddings.astype("float32")


# ----------------------------------------
# 2. Normalize vectors
# ----------------------------------------

faiss.normalize_L2(embeddings)


# ----------------------------------------
# 3. Get embedding dimension
# ----------------------------------------

dimension = embeddings.shape[1]

print("Embedding dimension:", dimension)


# ----------------------------------------
# 4. Create FAISS index
# ----------------------------------------

index = faiss.IndexFlatIP(dimension)


# ----------------------------------------
# 5. Add embeddings
# ----------------------------------------

index.add(embeddings)


print("Vectors stored:", index.ntotal)


# ----------------------------------------
# 6. Save index
# ----------------------------------------

faiss.write_index(
    index,
    "faiss.index"
)


print("\nFAISS index created successfully.")