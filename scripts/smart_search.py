#!/usr/bin/env python3
"""
Улучшенный поиск МДМ с индексацией по артикулам и парсингом характеристик.
"""

import json
import re
from difflib import SequenceMatcher
from collections import defaultdict
import os

DATA_PATH = '/home/clawd/.openclaw/skills/mdm-nomenclature/data/mdm_nomenclature.jsonl'

# Глобальные индексы
_article_index = {}  # артикул -> код МДМ
_characteristics_index = defaultdict(list)  # характеристика -> [(код, название)]
_all_records = []

def build_indexes():
    """Строит индексы для быстрого поиска."""
    global _article_index, _characteristics_index, _all_records
    
    if _all_records:
        return  # Уже построено
    
    print("Индексация базы МДМ...")
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            _all_records.append(rec)
            
            # Индекс по артикулу
            article = rec.get('article', '')
            if article:
                _article_index[article.lower()] = rec['code']
            
            # Индекс по характеристикам из названия
            name = rec.get('name', '').lower()
            chars = extract_characteristics(name)
            for key, val in chars.items():
                _characteristics_index[f"{key}:{val}"].append((rec['code'], rec['name']))
    
    print(f"  Проиндексировано: {len(_all_records):,} записей")
    print(f"  Уникальных артикулов: {len(_article_index)}")
    print(f"  Уникальных характеристик: {len(_characteristics_index)}")


def extract_characteristics(name: str) -> dict:
    """Извлекает характеристики из названия."""
    chars = {}
    name_lower = name.lower()
    
    # Полюса (1Р, 2Р, 3Р)
    poles_match = re.search(r'(\d)[рp]', name_lower)
    if poles_match:
        chars['полюса'] = poles_match.group(1)
    
    # Ток (40А, 16А)
    current_match = re.search(r'(\d+)а\b', name_lower)
    if current_match:
        chars['ток'] = current_match.group(1)
    
    # Напряжение (230квт, 400в)
    volt_match = re.search(r'(\d+)(?:квт|в)', name_lower)
    if volt_match:
        chars['напряжение'] = volt_match.group(1)
    
    # Отключающая способность (15кА, 10кА)
    ka_match = re.search(r'(\d+)(?:ка|ka)', name_lower)
    if ka_match:
        chars['отключающая_способность'] = ka_match.group(1)
    
    # Характеристика (C, D, B)
    curve_match = re.search(r'\b([cdb])\b', name_lower)
    if curve_match:
        chars['характеристика'] = curve_match.group(1).upper()
    
    # Серия (ВА 47-125, ВА-99)
    series_match = re.search(r'(ва[-]?\d+[-]?\d*)', name_lower)
    if series_match:
        chars['серия'] = series_match.group(1)
    
    # Артикул (mcb47125-1-40D, mcb4729-1-40C)
    article_match = re.search(r'(mcb\d+[\w-]+)', name_lower)
    if article_match:
        chars['артикул'] = article_match.group(1)
    
    # Размеры (710-710мм, 300х150мм)
    size_match = re.search(r'(\d+[хx-]\d+)\s*мм', name_lower)
    if size_match:
        chars['размер'] = size_match.group(1)
    
    # Толщина (s0,8мм, s1,5мм)
    thick_match = re.search(r's\s*(\d+[\.,]?\d*)\s*мм', name_lower)
    if thick_match:
        chars['толщина'] = thick_match.group(1)
    
    # ГОСТ
    gost_match = re.search(r'(гост\s*\d+-?\d*)', name_lower)
    if gost_match:
        chars['гост'] = gost_match.group(1)
    
    return chars


def search_by_article(query: str) -> list:
    """Поиск по артикулу (exact match)."""
    build_indexes()
    
    # Извлекаем артикул из запроса
    article_match = re.search(r'(mcb\d+[\w-]+)', query.lower())
    if article_match:
        article = article_match.group(1)
        code = _article_index.get(article.lower())
        if code:
            # Находим полную запись
            for rec in _all_records:
                if rec['code'] == code:
                    return [rec]
    
    return []


def search_by_characteristics(query: str, limit=10) -> list:
    """Поиск по характеристикам с scoring."""
    build_indexes()
    
    query_chars = extract_characteristics(query)
    if not query_chars:
        return []
    
    # Считаем совпадения по характеристикам
    code_scores = defaultdict(float)
    code_names = {}
    
    for key, val in query_chars.items():
        idx_key = f"{key}:{val}"
        matches = _characteristics_index.get(idx_key, [])
        for code, name in matches:
            code_scores[code] += 1.0  # Вес за точное совпадение характеристики
            code_names[code] = name
    
    # Добавляем fuzzy score для названия
    query_lower = query.lower()
    for rec in _all_records:
        code = rec['code']
        name = rec['name'].lower()
        
        # SequenceMatcher для нечёткого сравнения
        similarity = SequenceMatcher(None, query_lower, name).ratio()
        if similarity > 0.3:
            code_scores[code] += similarity * 0.5  # Дополнительный вес
            if code not in code_names:
                code_names[code] = rec['name']
    
    # Сортируем по score
    sorted_codes = sorted(code_scores.items(), key=lambda x: -x[1])
    
    # Возвращаем топ-N
    results = []
    for code, score in sorted_codes[:limit]:
        for rec in _all_records:
            if rec['code'] == code:
                rec_copy = rec.copy()
                rec_copy['_score'] = round(score, 2)
                results.append(rec_copy)
                break
    
    return results


def smart_search(query: str, limit=10) -> list:
    """
    Умный поиск:
    1. Сначала по артикулу (exact match)
    2. Потом по характеристикам
    3. fallback на fuzzy search
    """
    # Шаг 1: Поиск по артикулу
    results = search_by_article(query)
    if results:
        return results
    
    # Шаг 2: Поиск по характеристикам
    results = search_by_characteristics(query, limit)
    if results and results[0].get('_score', 0) >= 2.0:
        return results
    
    # Шаг 3: Fallback на fuzzy
    return fuzzy_fallback(query, limit)


def fuzzy_fallback(query: str, limit=10) -> list:
    """Классический fuzzy search через SequenceMatcher."""
    build_indexes()
    
    q = query.lower().strip()
    results = []
    
    for rec in _all_records:
        name = rec.get('name', '').lower()
        score = SequenceMatcher(None, q, name).ratio()
        if score >= 0.3:
            results.append((score, rec))
    
    results.sort(key=lambda x: -x[0])
    return [r for _, r in results[:limit]]


# Инициализация при импорте
build_indexes()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python smart_search.py 'название материала'")
        sys.exit(1)
    
    query = sys.argv[1]
    print(f"Поиск: \"{query}\"\n")
    
    results = smart_search(query, limit=10)
    
    if not results:
        print("Ничего не найдено")
        sys.exit(0)
    
    print(f"Найдено: {len(results)}\n")
    for i, r in enumerate(results, 1):
        score_info = f" (score: {r.get('_score', 'n/a')})" if '_score' in r else ""
        print(f"{i}. {r['name']}{score_info}")
        print(f"   Код МДМ: {r['code']}")
        print(f"   Класс: {r['class']} | Статус: {r['status']} | Ед: {r.get('unit', 'шт')}")
        print()
