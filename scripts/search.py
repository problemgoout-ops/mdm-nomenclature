#!/usr/bin/env python3
"""MDM Nomenclature Search — semantic search via OpenAI + pgvector with JSONL fallback."""

import json
import sys
import os
from difflib import SequenceMatcher

# --- OpenAI + pgvector ---
import psycopg2
from psycopg2 import pool
import openai

# PostgreSQL
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', 5432))
DB_NAME = os.getenv('DB_NAME', 'nomenclature_kb')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASS = os.getenv('DB_PASS', '')

# OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_API_KEY_HERE')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'mdm_nomenclature.jsonl')

_db_pool = None

def get_db_pool():
    global _db_pool
    if _db_pool is None:
        _db_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
            user=DB_USER, password=DB_PASS
        )
    return _db_pool

def get_embedding(text: str) -> list:
    """Get embedding via OpenAI API (1536 dim)."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI embedding error: {e}", file=sys.stderr)
        return None

def search_semantic(query, limit=10, status_filter=None, class_filter=None):
    """Semantic search via pgvector. Returns list of dicts or None if DB unavailable."""
    emb = get_embedding(query)
    if emb is None:
        return None

    try:
        conn = get_db_pool().getconn()
    except Exception as e:
        print(f"DB connection error: {e}", file=sys.stderr)
        return None

    try:
        cur = conn.cursor()
        sql = """
            SELECT code, name, class, attributes->>'status' as status, attributes->>'unit' as unit,
                   1 - (embedding <=> %s::vector) as similarity
            FROM mdm_reference
            WHERE embedding IS NOT NULL
        """
        params = [emb]
        if status_filter:
            sql += " AND attributes->>'status' = %s"
            params.append(status_filter)
        if class_filter:
            sql += " AND class = %s"
            params.append(class_filter)
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([emb, limit])

        cur.execute(sql, params)
        rows = cur.fetchall()
        return [{
            'code': row[0], 'name': row[1], 'class': row[2],
            'status': row[3] or 'Не указан', 'unit': row[4] or 'шт',
            'similarity': round(row[5] * 100, 1)
        } for row in rows]
    except Exception as e:
        print(f"Semantic search error: {e}", file=sys.stderr)
        return None
    finally:
        get_db_pool().putconn(conn)

def search_fuzzy(query, records, limit=10, status_filter=None, class_filter=None):
    """Fallback fuzzy search via JSONL + SequenceMatcher."""
    q = query.lower().strip()
    results = []
    for rec in records:
        if status_filter and rec.get('status') != status_filter:
            continue
        if class_filter and rec.get('class') != class_filter:
            continue
        name = rec.get('name', '').lower()
        if name == q:
            score = 1.0
        elif q in name:
            score = 0.8 + 0.2 * (len(q) / len(name))
        else:
            score = SequenceMatcher(None, q, name).ratio()
        if score >= 0.3:
            status = rec.get('status', '')
            if status == 'Эталон':
                score += 0.05
            elif status == 'Стандарт':
                score += 0.02
            results.append((score, rec))
    results.sort(key=lambda x: (-x[0], 0 if x[1].get('status') == 'Эталон' else (1 if x[1].get('status') == 'Стандарт' else 2)))
    return [r for _, r in results[:limit]]

def search_by_code(code, records, prefix=False, limit=10):
    """Exact or prefix search by code."""
    code = str(code).strip()
    if prefix:
        return [r for r in records if r.get('code', '').startswith(code)][:limit]
    return [r for r in records if r.get('code') == code][:limit]

def load_data(path=DATA_PATH):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def format_results(results, semantic=False):
    if not results:
        return "Ничего не найдено."
    lines = []
    for i, r in enumerate(results, 1):
        sim = f" ({r['similarity']}%)" if semantic and 'similarity' in r else ""
        lines.append(f"{i}. {r.get('name', '')}\n   Код: {r.get('code', '')} | Класс: {r.get('class', '')} | Статус: {r.get('status', '')} | Ед: {r.get('unit', '')}{sim}")
    return '\n'.join(lines)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MDM Nomenclature Search')
    parser.add_argument('--name', help='Search by name')
    parser.add_argument('--code', help='Search by code (exact)')
    parser.add_argument('--prefix', help='Search by code prefix', action='store_true')
    parser.add_argument('--status', help='Filter by status', choices=['Эталон', 'Стандарт', 'Архив'])
    parser.add_argument('--class', dest='class_filter', help='Filter by class')
    parser.add_argument('--limit', type=int, default=10, help='Max results')
    parser.add_argument('--fuzzy', action='store_true', help='Force fuzzy search (skip semantic)')
    args = parser.parse_args()

    records = load_data()

    if args.code:
        results = search_by_code(args.code, records, prefix=args.prefix, limit=args.limit)
        print(format_results(results))
        return

    if args.name:
        results = None
        if not args.fuzzy:
            results = search_semantic(args.name, limit=args.limit,
                                       status_filter=args.status,
                                       class_filter=args.class_filter)
        if results is None:
            print("⚠️ Semantic search unavailable, using fuzzy fallback", file=sys.stderr)
            results = search_fuzzy(args.name, records, limit=args.limit,
                                    status_filter=args.status,
                                    class_filter=args.class_filter)
            print(format_results(results))
        else:
            print(f"✅ Semantic search ({len(results)} results)", file=sys.stderr)
            print(format_results(results, semantic=True))

if __name__ == '__main__':
    main()
