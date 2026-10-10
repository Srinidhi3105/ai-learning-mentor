from openai import OpenAI
from dotenv import load_dotenv

from backend.config import EMBEDDING_MODEL


load_dotenv()

client = OpenAI()


def get_embedding(text):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


def get_embeddings(
    texts,
    batch_size=64
):

    all_embeddings = []


    for start in range(
        0,
        len(texts),
        batch_size
    ):

        batch = texts[
            start:start + batch_size
        ]


        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )


        ordered = sorted(
            response.data,
            key=lambda item: item.index
        )


        for item in ordered:

            all_embeddings.append(
                item.embedding
            )


    return all_embeddings