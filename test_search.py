#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/clawd/.openclaw/skills/mdm-nomenclature/scripts')
from search import search_fuzzy, load_records

records = load_records()
results = search_fuzzy("арматура A500C", records, limit=5)
print(json.dumps(results, ensure_ascii=False, indent=2))
