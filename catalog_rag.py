"""
pgvector Catalog rag

  - embed_catalog_items()     — batch-embed and upsert into pgvector
  - retrieve_catalog()        — cosine similarity search
  - make_catalog_retriever_tool() — factory for per-business LangChain @tool
  - embed_catalog_from_s3()   — download CSV/Excel from S3 and embed

"""


import io
from typing import List

import psycopg2
import psycopg2.extras

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import (
    DATABASE_URL,
    GOOGLE_API_KEY,
    GEMINI_EMBED_MODEL,
    EMBEDDING_DIMENSION,
    RAG_TOP_K,
    S3_BUCKET,
)

# Embedding

_embeddings = GoogleGenerativeAIEmbeddings(
    model = GEMINI_EMBED_MODEL,
    google_api_key = GOOGLE_API_KEY
)

# DB Helpers
def _get_conn():
    """Open a new psycopg2 connection. Caller must close it."""
    return psycopg2.connect(DATABASE_URL)

def _vec_to_pg(vector: List[float])->str:
    """Convert a Python float list to PostgreSQL vector literal '[x,y,...]'."""
    return "[" + ",".join(f"{v:8f}" for v in vector) + "]"

# Core functions

def embed_text(text:str)->List[float]:
    """Embed a single with Gemini text-embedding-004 (768 dims.)"""
    return _embeddings.embed_query(text)

def embed_catalog_items(business_id:str, items:List[dict])->int:
    """
    Batch-embed a list of product dicts and upsert into catalog_items.
    Old catalog for this business_id is deleted first.
 
    Each item dict should have:
      name (required), description, price, metadata (optional dict)
 
    Returns number of rows inserted.
    """
    if not items:
        return 0
    conn = _get_conn()

    try:
        with conn.cursor() as cur:
            # Clear old catalog for this business
            cur.execute(
                "DELETE FROM catalog_items WHERE business_id = %s",(business_id,)
            )
            count = 0
            for item in items:
                name = str(item.get("name") or "")
                desc = str(item.get("description") or "")
                price = item.get("price")
                metadata = item.get("metadata") or {}

                # embeddable text for product of catalog
                embeddable = f"{name}. {desc}. Price: {price} INR.".strip()

                vector = embed_text(embeddable)
                vec_literal = _vec_to_pg(vector)

                cur.execute(
                    """
                    INSERT INTO catalog_items
                      (business_id, name, description, price, currency, metadata, embedding)
                    VALUES (%s, %s, %s, %s, 'INR', %s, %s::vector)
                    """,
                    (
                        business_id,
                        name,
                        desc,
                        float(price) if price is not None else None,
                        psycopg2.extras.Json(metadata),
                        vec_literal,
                    ),
                )

                count +=1
            
            conn.commit()
            return count

    finally:
        conn.close()


def retrieve_catalog(business_id:str, query:str, top_k:int = RAG_TOP_K)->List[dict]:
    """
    Semantic search over catelog_items using cosine distance (<==>).
    Returns up to 'top_k' matching products, ordered by similarity.
    """

    query_vec = embed_text(query)
    vec_literal = _vec_to_pg(query_vec)

    conn = _get_conn()

    try:
        with conn.cursor(cursor_factory = psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                 """
                SELECT
                    name,
                    description,
                    price,
                    currency,
                    metadata,
                    ROUND((1 - (embedding <=> %s::vector))::numeric, 3) AS similarity
                FROM catalog_items
                WHERE business_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_literal, business_id, vec_literal, top_k),
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


# tool factory

def make_catalog_retriever_tool(business_id:str):
    """
    Create a LangChain @tool bound to as specific business_id.
    product_agent calls this to search the merchant's catalog.
    """

    @tool
    def catalog_retriever(query:str)->str:
        """
        Search the product catalog for items matching the customer's query.
        Use this whenever the customer asks about products, prices, or availability.
 
        Args:
            query: The customer's question about products (in any language)
 
        Returns:
            Formatted product matches as a WhatsApp-ready string
        """

        results = retrieve_catalog(business_id, query, top_k=RAG_TOP_K)

        if not results:
            return "No matching products found in the catalog for this query."
        
        lines = []
        for r in results:
            price_str = f"₹{r['price']:.0f}" if r.get("price") else "Price on request"
            desc = (r.get("description") or "")[:120]
            lines.append(f"• *{r['name']}* — {price_str}")
            if desc:
                lines.append(f" _{desc}_")
        
        return "Matching products:\n" + "\n".join(lines)
    return catalog_retriever


# s3 bases catelog loader
def embed_catalog_from_s3(business_id: str, s3_key:str)->int:
    """
    Download a catalog CSV or Excel file from S3, embed it, and store in pgvector.
    Called by dukan-api after a merchant uploads their product file.
    Returns number of items embedded.
    """

    import pandas as pd
    import boto3
    import os

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "ap-south-1"))
    obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    raw_bytes = obj["Body"].read()
    
    # parse based on file extension
    lower_key = s3_key.lower()
    if lower_key.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(raw_bytes))
    elif lower_key.endswith((".xlsx",".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes))
    else:
        raise ValueError(f"Unsupported file type for key: {s3_key}")

    # Normalise column names to lowercase

    df.columns = [c.strip().lower() for c in df.columns]
    
    items: List[dict] = []
    for _,row in df.iterrows():
        item: dict = {
            "name" : str(row.get("name") or ""),
            "description" : str(row.get("description") or row.get("desc") or ""),
            "price": row.get("price") or row.get("mrp") or 0,
            "metadata" : {}
        }

        # Stash extra column as meta data
        skip_cols = {"name", "description", "desc","price","mrp"}
        for col in df.columns:
            if col not in skip_cols:
                item["metadata"][col] = str(row.get(col,""))
        if item["name"]:
            items.append(item)
    
    return embed_catalog_items(business_id,items)


def embed_catalog_from_local_csv(business_id:str, filepath:str) -> int:
    """Load a local CSV file and embed. Used in development and seeding scripts."""
    import pandas as pd

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    items =[]
    for _, row in df.iterrows():
        item = {
            "name" : str(row.get("name") or ""),
            "description" : str(row.get("description") or ""),
            "price" : row.get("price") or 0,
            "metadata":{
                col : str(row.get(col,""))
                for col in df.columns
                if col not in {"name", "description", "price"}
            },

        }
        if item["name"]:
            items.append(item)
    
    return embed_catalog_items(business_id, items)

