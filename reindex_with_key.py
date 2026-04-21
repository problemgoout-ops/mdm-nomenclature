#!/usr/bin/env python3
"""Переиндексация с OpenAI API - требует OPENAI_API_KEY"""
import json
import os
import sys

# Проверяем ключ
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("❌ Ошибка: OPENAI_API_KEY не установлен", file=sys.stderr)
    print("Запусти: export OPENAI_API_KEY='sk-...' && python3 reindex_with_key.py", file=sys.stderr)
    sys.exit(1)

import psycopg2
import openai

client = openai.OpenAI(api_key=OPENAI_API_KEY)

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '')

DATA_PATH = 'data/mdm_nomenclature.jsonl'

def get_embedding(text):
    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"  OpenAI error: {e}", file=sys.stderr)
        return None

print("=" * 50, file=sys.stderr)
print("Переиндексация МДМ через OpenAI API", file=sys.stderr)
print("=" * 50, file=sys.stderr)

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

records = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line.strip()))

cur.execute("TRUNCATE TABLE mdm_reference")
conn.commit()
print(f"✓ Загружено {len(records)} записей", file=sys.stderr)

total = len(records)
success = 0

print("Начинаю генерацию эмбеддингов...", file=sys.stderr)

for i, rec in enumerate(records):
    emb = get_embedding(rec.get('name', ''))
    if emb:
        success += 1
    
    cur.execute(
        """INSERT INTO mdm_reference (code, name, class, attributes, embedding) 
           VALUES (%s, %s, %s, %s, %s)""",
        (rec.get('code'), rec.get('name'), rec.get('class'),
         json.dumps({'status': rec.get('status'), 'unit': rec.get('unit'), 
                    'article': rec.get('article'), 'gt': rec.get('gt')}), emb)
    )
    conn.commit()
    
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{total} done, embeddings OK: {success}", file=sys.stderr)

print(f"\n✅ Готово! Всего: {total}, с эмбеддингами: {success}", file=sys.stderr)

cur.execute("SELECT COUNT(*), COUNT(embedding) FROM mdm_reference")
count, with_emb = cur.fetchone()
print(f"✓ В БД: {count} записей, {with_emb} с эмбеддингами", file=sys.stderr)

cur.close()
conn.close()
