from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from RAGS.rag_core import answer as rag_answer
from .models.models import ChatRequest, ChatResponse, SourceItem

app = FastAPI(
    title="RAG Courses API",
    description="API pour questionner le modèle RAG sur les cours (Postgres + OpenAI).",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Endpoint principal utilisé par Flutter.
    """
    text_answer, rows = rag_answer(req.question, k=req.k or 5)

    sources = [
        SourceItem(
            source=src,
            modality=modality,
            score=float(score),
            chunk=chunk,
        )
        for src, chunk, modality, score in rows
    ]

    return ChatResponse(
        answer=text_answer,
        sources=sources,
    )
