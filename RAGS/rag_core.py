from database.db import get_conn
from utils.openai_utils import embed_text
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def retrieve(query: str, k: int = 5):
    q_emb = embed_text(query)
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                source,
                chunk,
                modality,
                1 - (embedding <=> %s::vector) AS score
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (q_emb, q_emb, k),
        )
        return cur.fetchall()


def answer(query: str, k: int = 5):
    rows = retrieve(query, k=k)
    context = "\n\n".join([f"[{m}] {c}" for _, c, m, _ in rows])

    prompt = f"""
Tu es un assistant RAG multimodal.
Utilise STRICTEMENT le contexte pour répondre.

Contexte:
{context}

Question:
{query}

Réponse:
"""

    resp = client.responses.create(
        model="gpt-5",
        input=prompt,
    )
    return resp.output_text, rows
