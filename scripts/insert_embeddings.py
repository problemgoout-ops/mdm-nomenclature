#!/usr/bin/env python3
"""Insert embeddings for new records into mdm_reference."""
import json, os, sys, time, math
import psycopg2
from psycopg2 import pool
import openai
import openpyxl

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
if not OPENAI_API_KEY:
    env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            OPENAI_API_KEY = f.read().strip()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 5432))
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '')

BATCH_SIZE = 50
EMBEDDING_MODEL = "text-embedding-3-small"
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH = os.path.join(SKILL_DIR, 'data', 'mdm_nomenclature.jsonl')

# Load existing PG codes
conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()
cur.execute("SELECT code FROM mdm_reference WHERE embedding IS NOT NULL")
pg_emb_codes = set(row[0] for row in cur.fetchall())
print(f"Records in PG with embeddings: {len(pg_emb_codes)}")

# Load JSONL, find records without embeddings
missing = []
with open(JSONL_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            if rec['code'] not in pg_emb_codes:
                missing.append(rec)

print(f"Records needing embeddings: {len(missing)}")
if not missing:
    print("All done!"); sys.exit(0)

# Insert in batches with per-transaction commit
inserted = 0
errors = 0
for i in range(0, len(missing), BATCH_SIZE):
    batch = missing[i:i+BATCH_SIZE]
    texts = [r['name'] for r in batch]
    
    try:
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        embeddings = [d.embedding for d in resp.data]
    except Exception as e:
        print(f"  Embedding error at batch {i}-{i+len(batch)}: {e}")
        time.sleep(5)
        continue

    # Each record in its own transaction to avoid cascading failure
    for j, rec in enumerate(batch):
        try:
            cur.execute("BEGIN")
            cur.execute("""
                INSERT INTO mdm_reference (code, name, class, embedding, attributes)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb)
                ON CONFLICT (code) DO UPDATE SET
                    name = EXCLUDED.name,
                    class = EXCLUDED.class,
                    embedding = EXCLUDED.embedding,
                    attributes = EXCLUDED.attributes
            """, (
                rec['code'], rec['name'], rec['class'], embeddings[j],
                json.dumps({'status': rec['status'], 'unit': rec['unit'], 
                          'class_code': rec.get('class_code',''), 'article': rec.get('article',''),
                          'gt': rec.get('gt','')}, ensure_ascii=False)
            ))
            cur.execute("COMMIT")
            inserted += 1
        except Exception as e:
            cur.execute("ROLLBACK")
            errors += 1
            if errors <= 5:
                print(f"  Insert error for {rec['code']}: {e}")

    if (i // BATCH_SIZE) % 10 == 0:
        print(f"  Progress: {inserted}/{len(missing)} inserted, {errors} errors")
    time.sleep(0.2)  # Rate limit buffer

print(f"\nDone! Inserted: {inserted}, Errors: {errors}, Total missing: {len(missing)}")
cur.close(); conn.close()
