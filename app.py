from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

question = input("Ask your question: ")

response = client.responses.create(
    model="gpt-5.6",
    instructions="You are an AI tutor. Explain concepts simply and clearly.",
    input=question
)

print("\nAI Tutor:\n")
print(response.output_text)