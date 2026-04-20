#!/usr/bin/env python3
"""Гибридный поиск: семантический + keyword фильтрация."""

import os
import sys
import re
import psycopg2
from psycopg2 import pool
import openai
from typing import List, Dict

# OpenAI API
OPENAI_API_KEY = "YOUR_API_KEY_HERE"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'nomenclature_kb',
    'user': 'clawd',
    'password': ''
}

_db_pool = None

def get_db_conn():
    global _db_pool
    if _db_pool is None:
        _db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
    return _db_pool.getconn()

def extract_keywords(query: str) -> Dict[str, List[str]]:
    """Извлечь ключевые слова из запроса."""
    query_lower = query.lower()
    
    # Паттерны для номенклатуры
    patterns = {
        'class': r'(арматура|кабель|воздуховод|труба|клапан|вентилятор|плитка|дверь|окно)',
        'marka': r'[АВЕКМНОРСТУХ][-\s]?\d+[СКМН]?',
        'diametr': r'[DdД][\s]?\d+',
        'gost': r'\d{4,5}[-\s]?\d{0,4}',
        'razmer': r'\d+х\d+',
        'dlina': r'[LlДд][\s]?\d+[.,]?\d*\s?[мm]'
    }
    
    keywords = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, query, re.IGNORECASE)
        if matches:
            keywords[key] = matches
    
    return keywords

def search_by_keywords(keywords: Dict[str, List[str]], limit: int = 20) -> List[Dict]:
    """Поиск по ключевым словам (ILIKE)."""
    conn = get_db_conn()
    
    conditions = []
    for key, values in keywords.items():
        for val in values:
            val_clean = re.sub(r'[^\w\d]', '', val)
            if len(val_clean) >= 2:
                conditions.append(f"name ILIKE '%{val_clean}%'")
    
    if not conditions:
        return []
    
    # OR между всеми условиями
    where_clause = ' OR '.join(conditions[:10])  # Макс 10 условий
    
    sql = f"""
        SELECT code, name, class,
               CASE 
                   WHEN {' AND '.join([c.replace('ILIKE', 'ILIKE').split('ILIKE')[0] + 'ILIKE ' + c.split('ILIKE')[1] for c in conditions[:3]])} THEN 100
                   ELSE 80 
               END as score
        FROM mdm_reference
        WHERE {where_clause}
        ORDER BY score DESC, name
        LIMIT %s
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()
            return [{
                'code': row[0],
                'name': row[1],
                'class': row[2],
                'similarity': row[3],
                'source': 'keyword'
            } for row in rows]
    except Exception as e:
        print(f"[KEYWORD ERROR] {e}", file=sys.stderr)
        return []
    finally:
        _db_pool.putconn(conn)

def search_semantic(query: str, limit: int = 10) -> List[Dict]:
    """Семантический поиск через OpenAI."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_vec = response.data[0].embedding
    
    conn = get_db_conn()
    sql = """
        SELECT code, name, class,
               1 - (embedding <=> %s::vector) as similarity
        FROM mdm_reference
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    
    try:
        with conn.cursor() as cur:
            cur.execute(sql, [query_vec, query_vec, limit])
            rows = cur.fetchall()
            return [{
                'code': row[0],
                'name': row[1],
                'class': row[2],
                'similarity': round(row[3] * 100, 1),
                'source': 'semantic'
            } for row in rows]
    finally:
        _db_pool.putconn(conn)

def search_hybrid(query: str, limit: int = 10) -> List[Dict]:
    """Гибридный поиск: keyword + semantic."""
    keywords = extract_keywords(query)
    
    # 1. Пробуем keyword
    keyword_results = search_by_keywords(keywords, limit=limit*2)
    
    # 2. Если нашли хорошие keyword-результаты (score >= 95), возвращаем их
    high_score = [r for r in keyword_results if r['similarity'] >= 95]
    if len(high_score) >= 3:
        return high_score[:limit]
    
    # 3. Иначе добавляем семантический поиск
    semantic_results = search_semantic(query, limit=limit*2)
    
    # 4. Объединяем, убираем дубликаты
    seen_codes = set()
    combined = []
    
    for r in keyword_results + semantic_results:
        if r['code'] not in seen_codes:
            seen_codes.add(r['code'])
            combined.append(r)
            if len(combined) >= limit:
                break
    
    return combined

def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "арматура А240 D12"
    results = search_hybrid(query, limit=5)
    
    print(f"Запрос: '{query}'")
    print(f"Найдено: {len(results)} результатов\n")
    
    for r in results:
        print(f"{r['code']} | {r['name'][:70]}... | {r['similarity']}% [{r['source']}]")

if __name__ == '__main__':
    main()
