"""Gözetimli öğrenme: müşteri önümüzdeki dönemde tekrar alışveriş yapar mı?

Zamana göre kurulur, sızıntı yoktur:
  - Kesim tarihi = verinin son günü - `horizon_days` (varsayılan 180 gün).
  - Özellikler YALNIZCA kesimden önceki alışverişlerden hesaplanır.
  - Etiket (1/0) = müşteri kesimden sonraki `horizon_days` içinde en az bir alışveriş yaptı mı?
Modeller: lojistik regresyon, rastgele orman. Referans: yalnızca "son alışveriş ne kadar önce" bilen basit model.
Ölçüt: ROC-AUC (0,5 = tesadüf, 1 = kusursuz), ortalama kesinlik (PR-AUC), kesinlik/duyarlılık.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config
from src.features import build_customer_features, share_columns


def build_dataset(df: pd.DataFrame, horizon_days: int = 180) -> tuple[pd.DataFrame, pd.Series, pd.Timestamp]:
    """Kesimden önceki özellikler (X) ve kesimden sonraki 'geri geldi mi' etiketi (y)."""
    cutoff = df["transaction_date"].max() - pd.Timedelta(days=horizon_days)
    before = df[df["transaction_date"] <= cutoff]
    after = df[df["transaction_date"] > cutoff]
    X = build_customer_features(before, ref_date=cutoff + pd.Timedelta(days=1))
    g = before.groupby("user_id")
    X["ort_fatura_tutari"] = X["monetary"] / X["frequency"].clip(lower=1)
    X["musteri_suresi_gun"] = (cutoff - g["transaction_date"].min()).dt.days
    X["ort_indirim"] = g["bill_discount"].mean()
    X["kategori_cesidi"] = g["category"].nunique()
    y = pd.Series(X.index.isin(set(after["user_id"])).astype(int), index=X.index, name="geri_geldi")
    return X, y, cutoff


def _metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "kesinlik": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "duyarlilik": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "karisiklik": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }


def train_and_evaluate(df: pd.DataFrame, horizon_days: int = 180, max_train: int = 200_000,
                       seed: int = config.RANDOM_STATE) -> dict:
    X, y, cutoff = build_dataset(df, horizon_days)
    if y.nunique() < 2 or len(y) < 100:
        return {"not": "Model için yeterli veri veya iki sınıf yok."}
    if len(X) > max_train:                                 # büyük veride hız için rastgele örnek
        keep = np.random.default_rng(seed).choice(len(X), size=max_train, replace=False)
        X, y = X.iloc[keep], y.iloc[keep]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    scaler = StandardScaler().fit(Xtr)
    # Referans model: yalnızca son alışverişten bu yana geçen süre
    base = LogisticRegression(max_iter=1000, class_weight="balanced").fit(scaler.transform(Xtr)[:, [list(X.columns).index("recency_days")]], ytr)
    p_base = base.predict_proba(scaler.transform(Xte)[:, [list(X.columns).index("recency_days")]])[:, 1]

    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=seed,
                                class_weight="balanced_subsample").fit(Xtr, ytr)
    p_rf = rf.predict_proba(Xte)[:, 1]

    cols = list(X.columns)
    importance = pd.DataFrame({
        "rf_onem": rf.feature_importances_,
        "lojistik_katsayi": lr.coef_[0],                     # standartlaştırılmış özellikler: işaret = yön
    }, index=cols).sort_values("rf_onem", ascending=False)
    curves = {n: roc_curve(yte, p) for n, p in [("Referans (sadece son alışveriş)", p_base),
                                                 ("Lojistik regresyon", p_lr), ("Rastgele orman", p_rf)]}
    return {
        "ozet": {
            "kesim_tarihi": str(cutoff.date()), "ufuk_gun": horizon_days,
            "n_egitim": int(len(Xtr)), "n_test": int(len(Xte)),
            "geri_gelme_orani": round(float(y.mean()), 4),
        },
        "referans": _metrics(yte, p_base),
        "lojistik_regresyon": _metrics(yte, p_lr),
        "rastgele_orman": _metrics(yte, p_rf),
        "onem": importance, "roc_egrileri": curves,
    }
