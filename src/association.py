"""Apriori ile birliktelik kuralları ("X alanlar Y de aldı").

Kütüphane kullanmadan, adım adım yazıldı; böylece algoritmanın nasıl çalıştığını görebilirsin.

Kavramlar:
  support(A)        = A'yı içeren sepet sayısı / toplam sepet sayısı
  confidence(A->B)  = support(A ve B) / support(A)      "A alanların kaçı B de aldı?"
  lift(A->B)        = confidence(A->B) / support(B)     1'den büyükse tesadüften güçlü ilişki
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import pandas as pd


def baskets_from_df(df: pd.DataFrame, level: str = "category") -> list[set[str]]:
    """Her fatura (bill_id) bir sepettir. level: 'category' veya 'item_id'."""
    return [set(v) for v in df.groupby("bill_id")[level].agg(list)]


def frequent_itemsets(baskets: list[set[str]], min_support: float = 0.01, max_len: int = 3):
    """Apriori: önce sık geçen tekliler, sonra bunlardan ikililer, üçlüler... (seviye seviye)."""
    n = len(baskets)
    min_count = max(1, math.ceil(min_support * n))

    singles = Counter(item for b in baskets for item in set(b))
    current = {frozenset([i]): c for i, c in singles.items() if c >= min_count}
    all_freq = dict(current)

    k = 2
    while current and k <= max_len:
        prev = set(current)
        items = sorted({i for s in prev for i in s})
        # Aday üretimi: bir önceki seviyenin TÜM alt kümeleri sıksa aday olabilir (Apriori ilkesi)
        candidates = set()
        for a in prev:
            for i in items:
                if i in a:
                    continue
                cand = a | {i}
                if len(cand) == k and all(frozenset(sub) in prev for sub in combinations(cand, k - 1)):
                    candidates.add(cand)
        counts: Counter = Counter()
        for b in baskets:
            present = sorted(set(b) & set(items))
            if len(present) < k:
                continue
            for combo in combinations(present, k):
                fs = frozenset(combo)
                if fs in candidates:
                    counts[fs] += 1
        current = {s: c for s, c in counts.items() if c >= min_count}
        all_freq.update(current)
        k += 1
    return all_freq, n


def association_rules(freq: dict, n_baskets: int, min_confidence: float = 0.1, min_lift: float = 1.0) -> pd.DataFrame:
    """Sık kümelerden 'antecedent -> consequent' kurallarını çıkarır (tek öğeli sonuç)."""
    rows = []
    for itemset, count in freq.items():
        if len(itemset) < 2:
            continue
        for cons in itemset:
            ante = itemset - {cons}
            conf = count / freq[ante]
            lift = conf / (freq[frozenset([cons])] / n_baskets)
            if conf >= min_confidence and lift >= min_lift:
                rows.append({
                    "antecedent": tuple(sorted(ante)),
                    "consequent": cons,
                    "support": count / n_baskets,
                    "confidence": conf,
                    "lift": lift,
                    "n_baskets": count,
                })
    cols = ["antecedent", "consequent", "support", "confidence", "lift", "n_baskets"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def mine_rules(df: pd.DataFrame, level: str = "category", min_support: float = 0.01,
               min_confidence: float = 0.1, min_lift: float = 1.0, max_len: int = 3) -> pd.DataFrame:
    baskets = baskets_from_df(df, level)
    freq, n = frequent_itemsets(baskets, min_support=min_support, max_len=max_len)
    return association_rules(freq, n, min_confidence=min_confidence, min_lift=min_lift)
