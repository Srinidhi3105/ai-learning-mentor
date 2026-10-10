from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

client = OpenAI()


# ----------------------------------------
# Structured evaluation result
# ----------------------------------------

class EvaluationResult(BaseModel):
    correctness_score: float = Field(ge=0, le=10)
    groundedness_score: float = Field(ge=0, le=10)
    passed: bool
    reason: str


# ----------------------------------------
# Evaluate one RAG answer
# ----------------------------------------

def evaluate_answer(
    question,
    context,
    generated_answer,
    expected_answer
):

    prompt = f"""
You are evaluating a Retrieval-Augmented Generation system.

Evaluate the generated answer using ONLY the information below.

Question:
{question}

Retrieved Context:
{context}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Evaluate two things:

1. Correctness
Does the generated answer correctly answer the question?

2. Groundedness
Is the generated answer supported by the retrieved context,
without adding unsupported claims?

Scores must be between 0 and 10.

Set passed=true only if:
- correctness_score >= 7
- groundedness_score >= 7
"""


    response = client.responses.parse(
        model="gpt-4.1-mini",
        input=prompt,
        text_format=EvaluationResult
    )

    return response.output_parsed


# ----------------------------------------
# Example test
# ----------------------------------------

question = "What is self-attention?"

context = """
Self-attention allows each token in a sequence
to consider other tokens when creating its representation.
"""

expected_answer = """
Self-attention allows tokens to examine other tokens
in the same sequence to understand relationships and context.
"""

generated_answer = """
Self-attention allows each token to look at other tokens
in the sequence so the model can understand relationships
between them.
"""


result = evaluate_answer(
    question,
    context,
    generated_answer,
    expected_answer
)


print("\nCorrectness:")
print(result.correctness_score)

print("\nGroundedness:")
print(result.groundedness_score)

print("\nPassed:")
print(result.passed)

print("\nReason:")
print(result.reason)