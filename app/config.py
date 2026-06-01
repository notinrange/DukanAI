import os
from dotenv import load_dotenv

load_dotenv()


GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBED_MODEL: str = os.getenv(
    "GEMINI_EMBEDDING_MODEL" , "models/text-embedding-004"
)


EMBEDDING_DIMENSION:int = 768

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