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


def _hits(recs_by_user: dict, relevant_by_user: dict, users: list) -> np.ndarray:
    """Her müşteri için isabet sayısı (aynı müşteri sırasıyla; yöntemler birbiriyle eşleştirilebilsin diye)."""
    return np.array([len(set(recs_by_user[u]) & relevant_by_user[u]) for u in users])


def bootstrap_ci(hit_flags: np.ndarray, n_boot: int = 2000, seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    """Ortalamanın %95 güven aralığı (müşteriler yeniden örneklenir)."""
    rng = np.random.default_rng(seed)
    n = len(hit_flags)
    means = hit_flags[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = config.RANDOM_STATE) -> dict:
    """A yöntemi ile B yönteminin isabet oranı farkı (A - B). Aynı müşteriler üzerinde eşleştirilmiş bootstrap.

    Güven aralığı 0'ı içermiyorsa fark tesadüf olmaktan çıkar (yaklaşık %95 güvenle).
    """
    d = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"fark": round(float(d.mean()), 4), "ci_alt": round(lo, 4), "ci_ust": round(hi, 4),
            "anlamli": bool(lo > 0 or hi < 0)}


def _score(hits: np.ndarray, k: int) -> dict:
    flags = (hits > 0).astype(float)
    lo, hi = bootstrap_ci(flags)
    return {
        "hit_rate@k": round(float(flags.mean()), 4),
        "hit_rate_ci95": [round(lo, 4), round(hi, 4)],
        "precision@k": round(float((hits / k).mean()), 4),
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
        hits = {name: _hits({u: fn(u) for u in users}, relevant, users) for name, fn in methods.items()}
        for name, h in hits.items():
            result["mevcut_musteri"][name] = _score(h, k)
        flags = {n: (h > 0) for n, h in hits.items()}
        result["mevcut_musteri"]["karsilastirma"] = {
            "hibrit_vs_populer": paired_bootstrap(flags["hibrit"], flags["populer"]),
            "benzer_urun_vs_populer": paired_bootstrap(flags["benzer_urun"], flags["populer"]),
            "hibrit_vs_segment_populer": paired_bootstrap(flags["hibrit"], flags["segment_populer"]),
        }

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
        cu = list(cold_rel)
        base = {u: [r.item_id for r in rec.recommend_new(p, k, use_segment=False).recs] for u, p in cold_prefs.items()}
        seg = {u: [r.item_id for r in rec.recommend_new(p, k).recs] for u, p in cold_prefs.items()}
        hb, hs = _hits(base, cold_rel, cu), _hits(seg, cold_rel, cu)
        result["yeni_musteri"]["kategori_populer"] = _score(hb, k)
        result["yeni_musteri"]["segment_tabanli"] = _score(hs, k)
        result["yeni_musteri"]["karsilastirma"] = {"segment_vs_kategori_populer": paired_bootstrap(hs > 0, hb > 0)}
    else:
        result["yeni_musteri"]["not"] = "Değerlendirme için yeterli yeni müşteri yok (en az 30 gerekir)."
    return result