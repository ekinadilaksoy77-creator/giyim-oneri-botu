"""Müşteri bazlı özellikler: RFM + kategori harcama payları."""
from __future__ import annotations

import pandas as pd

SHARE_PREFIX = "pay_"


def build_customer_features(df: pd.DataFrame, ref_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Her müşteri için tek satır üretir.

    - recency_days : son alışverişten bu yana geçen gün (küçük = yakın zamanda geldi)
    - frequency    : kaç farklı faturası var
    - monetary     : toplam harcama
    - pay_<KATEGORİ>: harcamanın yüzde kaçı o kategoriye gitti (satır toplamı 1)
    """
    ref_date = ref_date or (df["transaction_date"].max() + pd.Timedelta(days=1))
    g = df.groupby("user_id")
    rfm = pd.DataFrame({
        "recency_days": (ref_date - g["transaction_date"].max()).dt.days,
        "frequency": g["bill_id"].nunique(),
        "monetary": g["amount"].sum(),
    })
    spend = df.pivot_table(index="user_id", columns="category", values="amount", aggfunc="sum", fill_value=0.0)
    shares = spend.div(spend.sum(axis=1), axis=0)
    shares.columns = [f"{SHARE_PREFIX}{c}" for c in shares.columns]
    return rfm.join(shares)


def share_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c.startswith(SHARE_PREFIX)]
