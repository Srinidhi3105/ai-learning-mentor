import os
from dotenv import load_dotenv

load_dotenv(r"C:\ai-learning-mentor\.env")

print(os.getenv("OPENAI_API_KEY"))