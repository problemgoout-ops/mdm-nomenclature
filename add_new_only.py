#!/usr/bin/env python3
"""Добавление только новых записей из нового файла МДМ"""
import json
import os
import sys

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY не установлен", file=sys.stderr)
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
print("Добавление только новых записей", file=sys.stderr)
print("=" * 50, file=sys.stderr)

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

# Получаем уже загруженные коды
cur.execute("SELECT code FROM mdm_reference")
existing_codes = {row[0] for row in cur.fetchall()}
print(f"Уже в базе: {len(existing_codes)} записей", file=sys.stderr)

# Загружаем все записи из JSONL и фильтруем только новые
new_records = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line.strip())
        if rec.get('code') not in existing_codes:
            new_records.append(rec)

if not new_records:
    print("✅ Все записи уже в базе! Нечего добавлять.", file=sys.stderr)
    cur.close()
    conn.close()
    sys.exit(0)

print(f"Новых записей для добавления: {len(new_records)}", file=sys.stderr)

total = len(new_records)
success = 0

for i, rec in enumerate(new_records):
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
    
    if (i + 1) % 500 == 0 or (i + 1) == total:
        print(f"  {i+1}/{total} done, embeddings OK: {success}", file=sys.stderr)

print(f"\n✅ Готово! Добавлено: {total}, с эмбеддингами: {success}", file=sys.stderr)

cur.execute("SELECT COUNT(*), COUNT(embedding) FROM mdm_reference")
count, with_emb = cur.fetchone()
print(f"✓ Итого в БД: {count} записей, {with_emb} с эмбеддингами", file=sys.stderr)

cur.close()
conn.close()
