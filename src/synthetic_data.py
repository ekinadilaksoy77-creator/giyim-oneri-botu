"""Gerçek veri henüz indirilmediyse projeyi uçtan uca denemek için SENTETİK veri üretir.

Şema, Kaggle veri setiyle birebir aynıdır; böylece kodun geri kalanı değişmeden çalışır.
UYARI: Bu veri uydurmadır. Raporda sonuçları gerçek veri üzerinden vermelisin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# 4 gizli müşteri tipi: hangi kategorileri / renkleri tercih ediyorlar?
SEGMENT_PROFILES = {
    "klasik": {
        "weights": {"SHIRTS": .35, "TROUSERS": .25, "BELTS": .12, "JACKETS": .15, "SWEATERS": .05, "WALLETS": .05, "SOCKS": .03},
        "colours": ["NAVY", "INDIGO", "BLACK", "CHARCOAL", "LIGHT GREY"],
        "price_mult": 1.2,
    },
    "spor": {
        "weights": {"JOGGERS": .35, "JERSEYS": .25, "SHORTS": .2, "SOCKS": .1, "SHIRTS": .1},
        "colours": ["BLACK", "DARK GREY", "RED", "LIGHT GREY"],
        "price_mult": 0.9,
    },
    "dis_giyim": {
        "weights": {"JACKETS": .3, "SWEATERS": .25, "DENIM": .25, "FOOTWEAR": .15, "TROUSERS": .05},
        "colours": ["TAN", "KHAKI", "DARK DENIM", "CHARCOAL"],
        "price_mult": 1.1,
    },
    "aksesuar": {
        "weights": {"WALLETS": .35, "BELTS": .3, "SOCKS": .25, "SHIRTS": .1},
        "colours": ["BLACK", "TAN", "DARK GREY"],
        "price_mult": 0.8,
    },
}
SEGMENT_PROBS = [0.35, 0.30, 0.20, 0.15]

PRICE_RANGE = {
    "SHIRTS": (700, 2500), "JOGGERS": (900, 2000), "JERSEYS": (600, 1500), "JACKETS": (2500, 6000),
    "TROUSERS": (1200, 3000), "BELTS": (500, 1200), "SWEATERS": (1200, 2800), "SOCKS": (150, 400),
    "WALLETS": (600, 1500), "DENIM": (1500, 3500), "SHORTS": (700, 1500), "FOOTWEAR": (2000, 5000),
}
SIZED = {"SHIRTS", "TROUSERS", "JACKETS", "JERSEYS", "SWEATERS", "JOGGERS", "SHORTS", "DENIM"}
SINGULAR = {
    "SHIRTS": "shirt", "JOGGERS": "jogger", "JERSEYS": "jersey", "JACKETS": "jacket", "TROUSERS": "trouser",
    "BELTS": "belt", "SWEATERS": "sweater", "SOCKS": "socks", "WALLETS": "wallet", "DENIM": "denim jeans",
    "SHORTS": "shorts", "FOOTWEAR": "shoes",
}
ALL_COLOURS = sorted({c for p in SEGMENT_PROFILES.values() for c in p["colours"]} | {"WHITE", "GREEN"})
ZONES = ["Central", "North", "East", "South", "West"]


def generate(n_users: int = 6000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg_names = list(SEGMENT_PROFILES)
    rows = []
    bill_counter = 0
    day0 = pd.Timestamp("2022-01-01")

    for u in range(n_users):
        seg = SEGMENT_PROFILES[seg_names[rng.choice(len(seg_names), p=SEGMENT_PROBS)]]
        cats = list(seg["weights"])
        probs = np.array(list(seg["weights"].values()))
        probs = probs / probs.sum()
        user_size = rng.choice(["S", "M", "L", "XL"], p=[.15, .4, .3, .15])

        n_bills = 1 + rng.poisson(0.5) + (rng.poisson(3) if rng.random() < 0.1 else 0)
        days = np.sort(rng.integers(0, 365, size=n_bills))
        for d in days:
            bill_counter += 1
            bill_id = f"B{bill_counter:07d}"
            date = day0 + pd.Timedelta(days=int(d))
            on_root = rng.random() < 0.7
            zone = "ROOT" if on_root else rng.choice(ZONES)
            store = f"{zone}{rng.integers(1000, 9999)}"
            discount = float(rng.choice([0, 0, 0, 10, 20]))
            n_items = int(rng.choice([1, 2, 3, 4], p=[.5, .27, .15, .08]))
            for _ in range(n_items):
                cat = rng.choice(cats, p=probs) if rng.random() > 0.15 else rng.choice(list(PRICE_RANGE))
                colour = rng.choice(seg["colours"]) if rng.random() > 0.15 else rng.choice(ALL_COLOURS)
                lo, hi = PRICE_RANGE[cat]
                amount = round(float(rng.uniform(lo, hi) * seg["price_mult"]), 2)
                if cat in SIZED:
                    size = user_size if rng.random() > 0.2 else rng.choice(["S", "M", "L", "XL"])
                elif cat == "FOOTWEAR":
                    size = str(rng.choice([40, 41, 42, 43, 44]))
                else:
                    size = None
                inv_cat = cat
                # Gerçek veriyi taklit eden kirlilikler (temizleme kodunu test etmek için)
                if rng.random() < 0.05:
                    inv_cat = None            # kategori eksik -> description'dan türetilmeli
                if rng.random() < 0.10:
                    colour = None             # renk eksik -> NOT CAPTURED
                if rng.random() < 0.01:
                    amount = 0.0              # geçersiz tutar -> atılmalı
                rows.append({
                    "user_id": f"U{u:05d}",
                    "bill_id": bill_id,
                    "line_item_amount": amount,
                    "bill_discount": discount,
                    "transaction_date": date.strftime("%Y-%m-%d"),
                    "description": f"MENS {colour or ''} {SINGULAR[cat]}".strip(),
                    "inventory_category": inv_cat,
                    "colour": colour,
                    "size": size,
                    "zone_name": zone,
                    "store_name": store,
                    "year": 2022,
                })
    return pd.DataFrame(rows)


def write_csv(path=None, **kwargs):
    path = path or config.RAW_DIR / "sentetik_giyim_satis_2022.csv"
    df = generate(**kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path, df
