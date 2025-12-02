from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    k: int | None = 5


class SourceItem(BaseModel):
    source: str
    modality: str
    score: float
    chunk: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
