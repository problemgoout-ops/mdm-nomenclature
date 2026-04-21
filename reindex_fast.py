#!/usr/bin/env python3
"""Быстрая переиндексация с batch embedding через Ollama"""
import json
import os
import psycopg2
from psycopg2.extras import execute_values
import requests
import sys

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '')

OLLAMA_URL = 'http://localhost:11434/api/embed'
EMBEDDING_MODEL = 'qwen3-embedding:4b'
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mdm_nomenclature.jsonl')

def get_embeddings_batch(texts):
    """Получение эмбеддингов для batch текстов"""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "input": texts},
            timeout=300
        )
        result = response.json()
        return result.get('embeddings', [None] * len(texts))
    except Exception as e:
        print(f"Batch error: {e}", file=sys.stderr)
        return [None] * len(texts)

print("=" * 50, file=sys.stderr)
print("Быстрая переиндексация МДМ (batch mode)", file=sys.stderr)
print("=" * 50, file=sys.stderr)

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
conn.autocommit = False
cur = conn.cursor()

print("✓ Подключено к PostgreSQL", file=sys.stderr)

records = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line.strip()))
print(f"✓ Загружено {len(records)} записей", file=sys.stderr)

cur.execute("TRUNCATE TABLE mdm_reference")
conn.commit()
print("✓ Таблица очищена", file=sys.stderr)

BATCH_SIZE = 20  # меньше batch для стабильности
DB_BATCH = 200    # batch для PostgreSQL

inserted = 0
success_emb = 0
all_values = []

print(f"Начинаю обработку batch={BATCH_SIZE}...", file=sys.stderr)

for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i+BATCH_SIZE]
    texts = [rec.get('name', '')[:4000] for rec in batch]
    
    # Получаем эмбеддинги batch-ом
    embeddings = get_embeddings_batch(texts)
    
    for rec, emb in zip(batch, embeddings):
        if emb:
            success_emb += 1
        
        all_values.append((
            rec.get('code', ''),
            rec.get('name', ''),
            rec.get('class', ''),
            json.dumps({
                'status': rec.get('status', ''),
                'unit': rec.get('unit', ''),
                'article': rec.get('article', ''),
                'gt': rec.get('gt', '')
            }),
            emb
        ))
    
    # Вставляем в БД накопленное
    if len(all_values) >= DB_BATCH:
        try:
            execute_values(cur, 
                "INSERT INTO mdm_reference (code, name, class, attributes, embedding) VALUES %s",
                all_values[:DB_BATCH]
            )
            conn.commit()
            inserted += len(all_values[:DB_BATCH])
            all_values = all_values[DB_BATCH:]
        except Exception as e:
            print(f"DB error: {e}", file=sys.stderr)
            conn.rollback()
    
    if (i // BATCH_SIZE) % 10 == 0:
        print(f"...обработано {min(i+BATCH_SIZE, len(records))}/{len(records)} (эмбеддинги OK: {success_emb})", file=sys.stderr)

# Вставляем остаток
if all_values:
    try:
        execute_values(cur,
            "INSERT INTO mdm_reference (code, name, class, attributes, embedding) VALUES %s",
            all_values
        )
        conn.commit()
        inserted += len(all_values)
    except Exception as e:
        print(f"Final DB error: {e}", file=sys.stderr)

print(f"\n✓ Вставлено: {inserted}", file=sys.stderr)
print(f"✓ С эмбеддингами: {success_emb}", file=sys.stderr)

cur.execute("SELECT COUNT(*) FROM mdm_reference")
count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM mdm_reference WHERE embedding IS NOT NULL")
with_emb = cur.fetchone()[0]

print(f"\nИТОГ: {count} записей, {with_emb} с эмбеддингами", file=sys.stderr)

cur.close()
conn.close()
print("✅ Готово!", file=sys.stderr)
