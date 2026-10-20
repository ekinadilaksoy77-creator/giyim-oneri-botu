"""Öneri kalitesini ölçer: zamana göre ayırma (geçmişle eğit, geleceği tahmin et).

Metrikler:
  hit_rate@k   : önerilen k üründen en az biri müşterinin sonradan aldığı ürünlerden mi?
  precision@k  : önerilen k ürünün kaçı doğru çıktı?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.pipeline import build
from src.recommender import Preferences


def temporal_split(df: pd.DataFrame, quantile: float = 0.8):
    dates = df["transaction_date"].sort_values().reset_index(drop=True)
    cutoff = dates.iloc[int(quantile * (len(dates) - 1))]
    return df[df["transaction_date"] <= cutoff], df[df["transaction_date"] > cutoff], cutoff


def _score(recs_by_user: dict, relevant_by_user: dict, k: int) -> dict:
    hits = [len(set(r) & relevant_by_user[u]) for u, r in recs_by_user.items()]
    return {
        "hit_rate@k": round(float(np.mean([h > 0 for h in hits])), 4),
        "precision@k": round(float(np.mean([h / k for h in hits])), 4),
    }


def evaluate(df: pd.DataFrame, k: int = config.DEFAULT_TOP_K, max_users: int = 2000, max_cold: int = 3000,
             seed: int = config.RANDOM_STATE) -> dict:
    train, test, cutoff = temporal_split(df)
    art = build(train)
    rec = art.recommender
    catalog = set(rec.item_ids)
    result: dict = {"k": k, "kesim_tarihi": str(cutoff.date()), "egitim_satir": len(train), "test_satir": len(test)}

    # ---- 1) Mevcut müşteriler -------------------------------------------------
    test_items = test[test["item_id"].isin(catalog)].groupby("user_id")["item_id"].agg(set)
    users = [u for u in test_items.index if u in rec._user_pos]
    rng = np.random.default_rng(seed)
    if len(users) > max_users:
        users = list(rng.choice(users, size=max_users, replace=False))
    result["mevcut_musteri"] = {"n_musteri": len(users)}
    if users:
        relevant = {u: test_items[u] for u in users}
        methods = {
            "populer": lambda u: rec.popular_items(k, exclude={rec.item_ids[p] for p in rec.user_item[rec._user_pos[u]].indices}),
            "segment_populer": lambda u: [r.item_id for r in rec.recommend_existing(u, k, cf_weight=0.0).recs],
            "benzer_urun": lambda u: [r.item_id for r in rec.recommend_existing(u, k, cf_weight=1.0).recs],
            "hibrit": lambda u: [r.item_id for r in rec.recommend_existing(u, k).recs],
        }
        for name, fn in methods.items():
            result["mevcut_musteri"][name] = _score({u: fn(u) for u in users}, relevant, k)

    # ---- 2) Yeni müşteriler (soğuk başlangıç) ----------------------------------
    # Eğitimde hiç görmediğimiz, test döneminde >=2 faturası olan müşteriler.
    # İlk faturalarını "bota söylediği tercihler" sayıp sonraki alışverişlerini tahmin ederiz.
    new = test[~test["user_id"].isin(set(train["user_id"]))]
    nbills = new.groupby("user_id")["bill_id"].nunique()
    eligible = nbills[nbills >= 2].index
    result["yeni_musteri_uygun_toplam"] = int(len(eligible))
    if len(eligible) > max_cold:                      # büyük veride yavaşlamasın: rastgele örnekle
        eligible = rng.choice(eligible.to_numpy(), size=max_cold, replace=False)
    cold_rel, cold_prefs = {}, {}
    for u, g in new[new["user_id"].isin(set(eligible))].groupby("user_id"):
        g = g.sort_values("transaction_date")
        first_bill = g["bill_id"].iloc[0]
        first, later = g[g["bill_id"] == first_bill], g[g["bill_id"] != first_bill]
        rel = set(later["item_id"]) & catalog
        if rel:
            cold_rel[u] = rel
            cold_prefs[u] = Preferences(categories=sorted(set(first["category"])))
    result["yeni_musteri"] = {"n_musteri": len(cold_rel)}
    if len(cold_rel) >= 30:
        base = {u: [r.item_id for r in rec.recommend_new(p, k, use_segment=False).recs] for u, p in cold_prefs.items()}
        seg = {u: [r.item_id for r in rec.recommend_new(p, k).recs] for u, p in cold_prefs.items()}
        result["yeni_musteri"]["kategori_populer"] = _score(base, cold_rel, k)
        result["yeni_musteri"]["segment_tabanli"] = _score(seg, cold_rel, k)
    else:
        result["yeni_musteri"]["not"] = "Değerlendirme için yeterli yeni müşteri yok (en az 30 gerekir)."
    return result