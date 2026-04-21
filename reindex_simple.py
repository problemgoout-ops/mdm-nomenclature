#!/usr/bin/env python3
"""Простая переиндексация с контролем прогресса"""
import json
import os
import psycopg2
import requests
import sys

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '')

OLLAMA_URL = 'http://localhost:11434/api/embed'
EMBEDDING_MODEL = 'qwen3-embedding:4b'
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mdm_nomenclature.jsonl')

def get_embedding(text):
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBEDDING_MODEL, "input": text[:2000]},
            timeout=60
        )
        result = resp.json()
        if 'embeddings' in result and result['embeddings']:
            return result['embeddings'][0]
    except Exception as e:
        print(f"  Embed error: {e}", file=sys.stderr)
    return None

print("Старт переиндексации", file=sys.stderr)

conn = psycopg2.connect(host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

records = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line.strip()))

cur.execute("TRUNCATE TABLE mdm_reference")
conn.commit()
print(f"Загружено {len(records)} записей, таблица очищена", file=sys.stderr)

total = len(records)
success = 0
failed = 0

for i, rec in enumerate(records):
    emb = get_embedding(rec.get('name', ''))
    if emb:
        success += 1
    
    try:
        cur.execute(
            """INSERT INTO mdm_reference (code, name, class, attributes, embedding) 
               VALUES (%s, %s, %s, %s, %s)""",
            (rec.get('code'), rec.get('name'), rec.get('class'),
             json.dumps({'status': rec.get('status'), 'unit': rec.get('unit'), 
                        'article': rec.get('article'), 'gt': rec.get('gt')}), emb)
        )
        conn.commit()
    except Exception as e:
        print(f"  DB error: {e}", file=sys.stderr)
        failed += 1
        conn.rollback()
    
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{total} done, embeddings OK: {success}, failed: {failed}", file=sys.stderr)

print(f"\nГотово! Всего: {total}, с эмбеддингами: {success}, ошибок: {failed}", file=sys.stderr)
cur.close()
conn.close()
