import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY_ENV_VARS: tuple[str, ...] = (
    "GOOGLE_API_KEY_1",
    "GOOGLE_API_KEY_2",
    "GOOGLE_API_KEY_3",
)
GOOGLE_API_KEY: str = next(
    (key for name in GOOGLE_API_KEY_ENV_VARS if (key := os.getenv(name, "").strip())),
    "",
)
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
HF_EMBED_MODEL: str = os.getenv(
    "HF_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2" 
)


EMBEDDING_DIMENSION: int = 768

# Database Postgres + pgvector

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://dukanai:dukanai_dev@localhost:5432/dukanai"
)

# ── Redis (LangGraph checkpointer) ───────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
 
# ── AWS S3 (catalog uploads) ─────────────────────────────────────────────────
S3_BUCKET: str = os.getenv("S3_BUCKET", "dukanai-prod")
AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
 
# ── Agent behaviour ───────────────────────────────────────────────────────────
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
MAX_REPLY_CHARS: int = int(os.getenv("MAX_REPLY_CHARS", "1000"))
LLM_TEMPERATURE_CHAT: float = float(os.getenv("LLM_TEMPERATURE_CHAT", "0.3"))
LLM_TEMPERATURE_CLASSIFY: float = 0.0   # deterministic for intent classification
