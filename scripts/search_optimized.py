#!/usr/bin/env python3
"""Оптимизированный поиск: быстрый + качественный."""

import os
import sys
import hashlib
import psycopg2
from psycopg2 import pool
from functools import lru_cache
import openai
from typing import List, Dict
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

# OpenAI API
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'nomenclature_kb',
    'user': 'clawd',
    'password': ''
}

# Connection pool - pre-warmed
_db_pool = psycopg2.pool.SimpleConnectionPool(5, 20, **DB_CONFIG)

# Thread pool for OpenAI
_openai_executor = ThreadPoolExecutor(max_workers=3)

# In-memory cache для повторных запросов
_query_cache = {}
CACHE_SIZE = 2000

def get_cache_key(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()[:16]

def get_embedding_fast(query: str) -> List[float]:
    """Получить эмбеддинг с кэшем."""
    cache_key = get_cache_key(query)
    
    # Проверяем кэш
    if cache_key in _query_cache:
        return _query_cache[cache_key]
    
    # Новый запрос к OpenAI
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        timeout=5
    )
    embedding = response.data[0].embedding
    
    # Сохраняем в кэш (LRU простой)
    if len(_query_cache) >= CACHE_SIZE:
        _query_cache.pop(next(iter(_query_cache)))
    _query_cache[cache_key] = embedding
    
    return embedding

def search_optimized(query: str, limit: int = 5) -> List[Dict]:
    """Оптимизированный поиск (< 300ms)."""
    start = time.time()
    
    # 1. Эмбеддинг (с кэшем)
    query_vec = get_embedding_fast(query)
    embed_time = (time.time() - start) * 1000
    
    # 2. Быстрый поиск
    conn = _db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # Используем pre-computed запрос с placeholder
            cur.execute("""
                SELECT code, name, class, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM mdm_reference
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [query_vec, query_vec, limit])
            rows = cur.fetchall()
    finally:
        _db_pool.putconn(conn)
    
    total_time = (time.time() - start) * 1000
    
    return [{
        'code': row[0],
        'name': row[1][:70] + '...' if len(row[1]) > 70 else row[1],
        'class': row[2],
        'similarity': round(row[3] * 100, 1),
        'embed_ms': round(embed_time, 1),
        'total_ms': round(total_time, 1)
    } for row in rows]

def search_with_quality_check(query: str, limit: int = 5) -> Dict:
    """Поиск с оценкой качества."""
    results = search_optimized(query, limit)
    
    # Оценка качества
    top_score = results[0]['similarity'] if results else 0
    
    quality = "high" if top_score >= 85 else "medium" if top_score >= 70 else "low"
    confidence = "✅ Найдено" if top_score >= 85 else "⚠️ Проверьте" if top_score >= 70 else "❌ Возможно отсутствует"
    
    return {
        'query': query,
        'results': results,
        'quality': quality,
        'confidence': confidence,
        'time_ms': results[0]['total_ms'] if results else 0
    }

def main():
    if len(sys.argv) > 1:
        query = sys.argv[1]
    else:
        query = "арматура A500C D12"
    
    print(f"🔍 Поиск: '{query}'")
    print("-" * 60)
    
    result = search_with_quality_check(query)
    
    print(f"⚡ Время: {result['time_ms']}ms (эмбеддинг: {result['results'][0]['embed_ms']}ms)")
    print(f"📊 Качество: {result['quality']} | {result['confidence']}")
    print("-" * 60)
    
    for r in result['results'][:5]:
        status = "✅" if r['similarity'] >= 85 else "⚠️" if r['similarity'] >= 70 else "❌"
        print(f"{status} {r['code']} | {r['name'][:45]}... | {r['similarity']}%")

if __name__ == '__main__':
    main()
