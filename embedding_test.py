import numpy as np

embeddings = {
    "king": np.array([0.8, 0.2, 0.9]),
    "queen": np.array([0.79, 0.25, 0.88]),
    "car": np.array([0.1, 0.9, 0.2]),
}

print("King vector:")
print(embeddings["king"])

print("\nQueen vector:")
print(embeddings["queen"])

print("\nCar vector:")
print(embeddings["car"])

def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


king_queen = cosine_similarity(
    embeddings["king"],
    embeddings["queen"]
)

king_car = cosine_similarity(
    embeddings["king"],
    embeddings["car"]
)

print("\nKing vs Queen:")
print(king_queen)

print("\nKing vs Car:")
print(king_car)