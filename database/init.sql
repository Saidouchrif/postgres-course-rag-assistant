CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS documents CASCADE;

CREATE TABLE documents (
  id SERIAL PRIMARY KEY,
  source   TEXT,
  chunk    TEXT,
  modality TEXT,         -- "text" ou "image"
  embedding VECTOR(1536) -- dim = text-embedding-3-small
);

CREATE INDEX IF NOT EXISTS idx_documents_embedding
ON documents
USING ivfflat (embedding vector_cosine_ops);
