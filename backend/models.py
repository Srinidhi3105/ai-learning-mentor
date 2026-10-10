from pydantic import BaseModel, Field
from typing import Literal


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class StoredHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    source_pages: list[int] = Field(
        default_factory=list
    )


class QuestionRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=200
    )

    question: str = Field(
        min_length=1,
        max_length=3000
    )


class QuestionResponse(BaseModel):
    answer: str
    source_pages: list[int]
    search_query: str


class UploadResponse(BaseModel):
    message: str
    filename: str
    pages: int
    chunks: int
    vectors: int


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[StoredHistoryMessage]


class RerankItem(BaseModel):
    chunk_index: int
    score: float = Field(
        ge=0,
        le=10
    )


class RerankResult(BaseModel):
    results: list[RerankItem]