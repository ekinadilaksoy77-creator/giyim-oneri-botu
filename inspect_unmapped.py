"""Kategorisi belirlenemeyen satırları inceler: hangi ön ek (ilk kelime) kaç satır, örnek satırlar neler?

Kullanım (proje klasöründe):  python inspect_unmapped.py
"""
import sys

import pandas as pd

import config
from src import data_loader

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_columns", 20)

paths = data_loader.find_data_files()
print("Okunan dosyalar:", [p.name for p in paths])
raw = pd.concat([data_loader.read_raw(p) for p in paths], ignore_index=True)
raw["line_item_amount"] = pd.to_numeric(raw["line_item_amount"], errors="coerce")
raw = raw[raw["line_item_amount"] >= 1]

cat = data_loader.categorize(raw["inventory_category"], raw["description"])
un = raw[cat.isna()].copy()
print(f"\nKategorisi belirlenemeyen satır: {len(un):,} / {len(raw):,}")

un["onek"] = un["inventory_category"].fillna("(boş)").astype(str).str.split().str[0]
counts = un["onek"].value_counts()
print("\nİlk kelimeye (ön ek) göre en sık 30 grup:")
print(counts.head(30).to_string())

n_show = int(sys.argv[1]) if len(sys.argv) > 1 else 12
cols = ["inventory_category", "description", "colour", "size", "line_item_amount"]
for pref in counts.head(n_show).index:
    print(f"\n=== Ön ek: {pref}  ({counts[pref]:,} satır) — rastgele 3 örnek ===")
    print(un[un["onek"] == pref][cols].sample(min(3, counts[pref]), random_state=1).to_string(index=False))
