#!/usr/bin/env python3
"""Семантический поиск по базе номенклатуры МДМ через pgvector + bge-m3."""
import os
import sys
import json
import hashlib
from typing import List, Dict, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import psycopg2
from psycopg2 import pool

# Ollama API
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'bge-m3:latest')
EMBEDDING_TIMEOUT = int(os.getenv('EMBEDDING_TIMEOUT', '30'))

# PostgreSQL
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 5432))
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'clawd')
DB_PASS = os.getenv('DB_PASS', '')
DB_POOL_MIN = int(os.getenv('DB_POOL_MIN', '1'))
DB_POOL_MAX = int(os.getenv('DB_POOL_MAX', '10'))

_db_pool = None

def get_db_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = psycopg2.pool.SimpleConnectionPool(
            DB_POOL_MIN, DB_POOL_MAX,
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )
    return _db_pool

def get_db_conn():
    return get_db_pool().getconn()

def put_db_conn(conn):
    get_db_pool().putconn(conn)

@lru_cache(maxsize=1000)
def get_embedding_cached(text_hash: str, text: str):
    import requests
    url = f"{OLLAMA_HOST}/api/embeddings"
    payload = {"model": OLLAMA_MODEL, "prompt": text}
    try:
        resp = requests.post(url, json=payload, timeout=EMBEDDING_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

import openai

# OpenAI API (временно для поиска)
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

import openai

# OpenAI API для эмбеддингов поиска
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> List[float]:
    """Получить эмбеддинг через OpenAI API (1536 dim) для поиска."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_semantic(query: str, limit: int = 10) -> List[Dict]:
    query_vec = get_embedding(query)
    
    sql = """
        SELECT code, name, class, 
               1 - (embedding <=> %s::vector) as similarity
        FROM mdm_reference
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, [query_vec, query_vec, limit])
            rows = cur.fetchall()
            return [{
                'code': row[0], 'name': row[1], 'class': row[2],
                'similarity': round(row[3] * 100, 1)
            } for row in rows]
    finally:
        put_db_conn(conn)

def search_batch(queries: List[str], limit: int = 10) -> List[List[Dict]]:
    """Batch-поиск для множества запросов."""
    # Получаем эмбеддинги параллельно
    with ThreadPoolExecutor(max_workers=5) as executor:
        embeddings = list(executor.map(get_embedding, queries))
    
    conn = get_db_conn()
    try:
        sql = """
            SELECT code, name, class, classif_code, status, unit, article, gt,
                   1 - (embedding <=> %s::vector) as similarity
            FROM nomenclature_vectors
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        results = []
        for query_vec in embeddings:
            with conn.cursor() as cur:
                cur.execute(sql, [query_vec, query_vec, limit])
                rows = cur.fetchall()
                results.append([{
                    'code': row[0], 'name': row[1], 'class': row[2],
                    'classif_code': row[3], 'status': row[4], 'unit': row[5],
                    'article': row[6], 'gt': row[7], 'similarity': round(row[8] * 100, 1)
                } for row in rows])
        return results
    finally:
        put_db_conn(conn)

def main():
    if len(sys.argv) < 2:
        print("Usage: search_semantic.py <query> [--limit N]")
        sys.exit(1)
    
    query = sys.argv[1]
    limit = 10
    
    if '--limit' in sys.argv:
        limit = int(sys.argv[sys.argv.index('--limit') + 1])
    
    results = search_semantic(query, limit=limit)
    for r in results:
        print(f"{r['code']} | {r['name']} | {r['similarity']}%")

if __name__ == '__main__':
    main()
