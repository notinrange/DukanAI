"""
scripts/seed_catalog.py — Seed a dummy 10-product catalog for testing
──────────────────────────────────────────────────────────────────────
Creates a demo business + agent row, then embeds 10 products via
catalog_rag.embed_catalog_items() so product_agent can answer queries.

Usage:
    python scripts/seed_catalog.py

Output:
    Prints the business_id — copy it into your .env or test calls.
"""
import os
import sys
import uuid
import psycopg2
from pathlib import Path

# Allow importing app package from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://dukanai:dukanai_dev@localhost:5432/dukanai",
)

# ── 10-product demo catalog (ladies fashion — works well for Hindi tests) ─────
DEMO_BUSINESS_ID = "11111111-1111-1111-1111-111111111111"   # fixed UUID for dev

DEMO_PRODUCTS = [
    {"name": "Anarkali Kurti (Red)", "description": "Cotton Anarkali kurti, A-line silhouette, suitable for festivals and casual wear", "price": 799},
    {"name": "Anarkali Kurti (Blue)", "description": "Cotton Anarkali kurti in royal blue, embroidered neckline", "price": 849},
    {"name": "Palazzo Pants (Black)", "description": "Rayon palazzo pants, wide-leg fit, elastic waist, one size fits most", "price": 499},
    {"name": "Salwar Suit (Pink)", "description": "3-piece cotton salwar suit, includes kameez, salwar and dupatta. Size: S/M/L/XL", "price": 1299},
    {"name": "Lehenga Choli (Green)", "description": "Silk-blend lehenga choli set for weddings, fully stitched, heavy embroidery", "price": 2499},
    {"name": "Saree (Banarasi)", "description": "Pure Banarasi silk saree, gold zari weave, comes with unstitched blouse piece", "price": 4999},
    {"name": "Crop Top (White)", "description": "Cotton crop top, round neck, short sleeves, machine washable", "price": 349},
    {"name": "Denim Jacket (Blue)", "description": "Stretch denim jacket, slim fit, 2 chest pockets, sizes S to XXL", "price": 1199},
    {"name": "Ethnic Earrings (Gold)", "description": "Kundan jhumka earrings, gold-plated, traditional Rajasthani design", "price": 299},
    {"name": "Potli Bag (Multicolor)", "description": "Embroidered potli bag, drawstring closure, suitable for bridal and festive occasions", "price": 449},
]


def seed():
    print(f"\n🌱  DukanAI Catalog Seeder")
    print(f"   DATABASE_URL : {DATABASE_URL[:40]}...")
    print(f"   Business ID  : {DEMO_BUSINESS_ID}\n")

    # ── 1. Create demo business row ───────────────────────────────────────────
    try:
        conn = psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"❌  DB connect failed: {e}")
        sys.exit(1)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO businesses (id, name, category, owner_name, owner_phone, owner_email)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            DEMO_BUSINESS_ID,
            "Priya Fashion House",
            "fashion",
            "Priya Sharma",
            "9876543210",
            "priya@dukanai.test",
        ))

        # ── 2. Create demo agent config row ──────────────────────────────────
        cur.execute("""
            INSERT INTO agents (business_id, name, personality, work_always_on)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (DEMO_BUSINESS_ID, "Priya Fashion AI", "friendly", True))

        conn.commit()
    conn.close()
    print("✓  Business + agent rows created (or already exist)")

    # ── 3. Embed catalog via catalog_rag ─────────────────────────────────────
    print("⏳  Embedding 10 products with Gemini text-embedding-004 ...")
    print("   (First run: ~15–30 seconds for 10 API calls)\n")

    try:
        from app.rag.catalog_rag import embed_catalog_items
        n = embed_catalog_items(DEMO_BUSINESS_ID, DEMO_PRODUCTS)
        print(f"✅  {n} products embedded successfully!\n")
        print(f"📋  Test with:\n")
        print(f'    curl -s -X POST http://localhost:8000/agent/process \\')
        print(f'      -H "Content-Type: application/json" \\')
        print(f'      -d \'{{"message":"red kurti ka price kya hai?","customer_id":"919999999999","business_id":"{DEMO_BUSINESS_ID}"}}\'\n')
    except Exception as e:
        print(f"❌  Embedding failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    seed()
