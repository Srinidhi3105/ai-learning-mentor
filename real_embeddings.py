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


sentence1 = "I love dogs"
sentence2 = "I like puppies"
sentence3 = "The stock market crashed today"


embedding1 = get_embedding(sentence1)
embedding2 = get_embedding(sentence2)
embedding3 = get_embedding(sentence3)


print("Embedding dimensions:")
print(len(embedding1))


print("\nI love dogs vs I like puppies:")
print(cosine_similarity(embedding1, embedding2))


print("\nI love dogs vs The stock market crashed today:")
print(cosine_similarity(embedding1, embedding3))