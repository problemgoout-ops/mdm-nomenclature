#!/usr/bin/env python3
import json
import os
import psycopg2
from psycopg2.extras import execute_values
import requests
import sys

# Настройки
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '')

OLLAMA_URL = 'http://localhost:11434/api/embed'
EMBEDDING_MODEL = 'qwen3-embedding:4b'

DATA_PATH = 'data/mdm_nomenclature.jsonl'

def get_embedding_ollama(text):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "input": text[:4000]},
            timeout=30
        )
        result = response.json()
        if 'embeddings' in result and len(result['embeddings']) > 0:
            return result['embeddings'][0]
        return None
    except Exception as e:
        print(f"Embedding error: {e}", file=sys.stderr)
        return None

print("=" * 50, file=sys.stderr)
print("Переиндексация базы МДМ через Ollama", file=sys.stderr)
print(f"Модель: {EMBEDDING_MODEL}", file=sys.stderr)
print("=" * 50, file=sys.stderr)

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
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

print("Начинаю генерацию эмбеддингов...", file=sys.stderr)
batch_size = 50
inserted = 0
errors = 0
success_emb = 0

for i in range(0, len(records), batch_size):
    batch = records[i:i+batch_size]
    values = []
    for rec in batch:
        name = rec.get('name', '')
        embedding = get_embedding_ollama(name)
        if embedding:
            success_emb += 1
        
        values.append((
            rec.get('code', ''),
            name,
            rec.get('class', ''),
            json.dumps({
                'status': rec.get('status', ''),
                'unit': rec.get('unit', ''),
                'article': rec.get('article', ''),
                'gt': rec.get('gt', '')
            }),
            embedding
        ))
    
    try:
        execute_values(cur, "INSERT INTO mdm_reference (code, name, class, attributes, embedding) VALUES %s", values)
        conn.commit()
        inserted += len(batch)
    except Exception as e:
        print(f"Ошибка batch {i}: {e}", file=sys.stderr)
        errors += len(batch)
        conn.rollback()
    
    if (i // batch_size) % 10 == 0:
        print(f"...обработано {inserted}/{len(records)} (эмбеддинги: {success_emb})", file=sys.stderr)

print(f"\n✓ Вставлено: {inserted}, ошибок: {errors}", file=sys.stderr)
print(f"✓ С эмбеддингами: {success_emb}", file=sys.stderr)

cur.execute("SELECT COUNT(*) FROM mdm_reference")
count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM mdm_reference WHERE embedding IS NOT NULL")
with_emb = cur.fetchone()[0]

print(f"\nИТОГ: {count} записей, {with_emb} с эмбеддингами", file=sys.stderr)

cur.close()
conn.close()
print("✅ Переиндексация завершена!", file=sys.stderr)
