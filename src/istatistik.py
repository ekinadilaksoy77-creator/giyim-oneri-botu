"""Klasik istatistik: regresyon, ANOVA / Kruskal-Wallis ve ki-kare bağımsızlık testi.

Bu dosyadaki yöntemler veri madenciliği modellerini (k-means, Apriori) tamamlar:
  1) Aylık ciro regresyonu : ciro zamanla artıyor mu, mevsimsellik var mı, ileriyi ne kadar tahmin edebiliyoruz?
  2) Segmentler arası fark : k-means'in bulduğu gruplar harcamada gerçekten farklı mı? (ANOVA + Kruskal-Wallis)
  3) Segment x bölge       : segmentler bölgeye göre değişiyor mu? (ki-kare + Cramér's V)

Not: p-değeri çok büyük örneklemde (yüz binlerce müşteri) neredeyse her zaman çok küçük çıkar.
Bu yüzden her testin yanında ETKİ BÜYÜKLÜĞÜ (R², eta², Cramér's V) de verilir; yorum onunla yapılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ 1) Aylık ciro regresyonu
def monthly_revenue(df: pd.DataFrame) -> pd.Series:
    s = df.groupby(df["transaction_date"].dt.to_period("M"))["amount"].sum().sort_index()
    return s.astype(float)


def _design(t: np.ndarray, months: np.ndarray, seasonal: bool = True) -> np.ndarray:
    """Sabit + doğrusal trend (+ seasonal ise 11 ay kuklası; Ocak taban ay)."""
    if not seasonal:
        return np.column_stack([np.ones(len(t)), t])
    dummies = np.stack([(months == m).astype(float) for m in range(2, 13)], axis=1)
    return np.column_stack([np.ones(len(t)), t, dummies])


def _ols(X: np.ndarray, y: np.ndarray) -> dict:
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - p, 1)
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X)))
    tvals = beta / np.where(se > 0, se, np.nan)
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    return {"beta": beta, "se": se, "p": pvals, "r2": r2, "adj_r2": adj, "dof": dof}


def revenue_regression(df: pd.DataFrame, horizon: int = 6, holdout: int = 6) -> dict:
    """ciro_t = a + b*t + ay etkisi + hata.  Son `holdout` ayı saklayıp tahmin hatasını (MAPE) ölçer."""
    y = monthly_revenue(df)
    idx = y.index
    # Zaman ekseni takvime göre: araya ay eksik girerse (örn. kapalı dönem) trend kaymasın
    t = np.array([(p - idx[0]).n for p in idx], dtype=float)
    months = np.array([p.month for p in idx])
    eksik = [str(p) for p in pd.period_range(idx[0], idx[-1], freq="M") if p not in idx]
    seasonal = len(y) >= 24            # 13 katsayıyı tahmin etmek için en az iki yıl veri gerekir
    X = _design(t, months, seasonal)
    fit = _ols(X, y.to_numpy())
    out = {
        "n_ay": int(len(y)), "eksik_aylar": eksik, "mevsimsel_model": bool(seasonal),
        "trend_katsayisi_aylik": float(fit["beta"][1]),
        "trend_p": float(fit["p"][1]),
        "r2": float(fit["r2"]), "duzeltilmis_r2": float(fit["adj_r2"]),
        "ortalama_aylik_ciro": float(y.mean()),
    }
    fitted = pd.Series(X @ fit["beta"], index=idx)

    # ileri tahmin
    fut_idx = pd.period_range(idx[-1] + 1, periods=horizon, freq="M")
    t_f = np.array([(p - idx[0]).n for p in fut_idx], dtype=float)
    pred = pd.Series(_design(t_f, np.array([p.month for p in fut_idx]), seasonal) @ fit["beta"], index=fut_idx)

    # saklı (holdout) doğrulama: model son aylara hiç bakmadan kurulur
    out["holdout_ay"] = 0
    if seasonal and len(y) - holdout >= 24:     # eğitim kısmı da en az iki yıl olmalı
        tr = slice(0, len(y) - holdout)
        f2 = _ols(X[tr], y.to_numpy()[tr])
        p2 = X[len(y) - holdout:] @ f2["beta"]
        actual = y.to_numpy()[len(y) - holdout:]
        out["holdout_ay"] = int(holdout)
        out["holdout_mape"] = float(np.mean(np.abs((actual - p2) / actual)))
    names = ["sabit", "trend"] + ([f"ay_{m}" for m in range(2, 13)] if seasonal else [])
    coef = pd.DataFrame({"katsayi": fit["beta"], "std_hata": fit["se"], "p_degeri": fit["p"]}, index=names)
    return {"ozet": out, "katsayilar": coef, "gercek": y, "uyum": fitted, "tahmin": pred}


# ------------------------------------------------------------------ 2) Segmentler arası fark
def segment_spend_tests(features: pd.DataFrame, labels: pd.Series, min_group: int = 2) -> dict:
    """Müşteri başına log(toplam harcama) segmentlere göre farklı mı? ANOVA + Kruskal-Wallis + eta-kare.

    UYARI: Segmentler zaten harcamayı da kullanarak kuruldu; bu yüzden fark çıkması şaşırtıcı değildir.
    Test, farkın BÜYÜKLÜĞÜNÜ göstermek için raporlanır (hipotezi 'kanıtlamak' için değil).
    """
    d = pd.DataFrame({"y": np.log1p(features["monetary"]), "seg": labels.reindex(features.index)}).dropna()
    groups = [g["y"].to_numpy() for _, g in d.groupby("seg") if len(g) >= min_group]
    f, p = stats.f_oneway(*groups)
    h, p_kw = stats.kruskal(*groups)
    grand = d["y"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = float(((d["y"] - grand) ** 2).sum())
    return {
        "grup_sayisi": len(groups), "n": int(len(d)),
        "anova_F": float(f), "anova_p": float(p),
        "kruskal_H": float(h), "kruskal_p": float(p_kw),
        "eta_kare": float(ss_between / ss_total) if ss_total > 0 else float("nan"),
    }


# ------------------------------------------------------------------ 3) Segment x bölge (ki-kare)
def segment_zone_test(df: pd.DataFrame, labels: pd.Series) -> dict:
    """Müşterinin segmenti ile ilk alışveriş yaptığı bölge bağımsız mı? (Segmentler bölge bilgisiyle kurulmadı.)"""
    first = df.sort_values("transaction_date").drop_duplicates("user_id")[["user_id", "zone_name"]]
    first = first.assign(seg=first["user_id"].map(labels)).dropna(subset=["seg", "zone_name"])
    table = pd.crosstab(first["seg"], first["zone_name"])
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"not": "Ki-kare için en az 2 segment ve 2 bölge gerekir."}
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return {"n": int(n), "segment_sayisi": int(table.shape[0]), "bolge_sayisi": int(table.shape[1]),
            "ki_kare": float(chi2), "serbestlik": int(dof), "p": float(p), "cramer_v": v, "tablo": table}


def yorum_etki(value: float, kind: str) -> str:
    """Etki büyüklüğünü sözle anlatır (Cohen'in genel eşikleri)."""
    th = {"eta": (0.01, 0.06, 0.14), "v": (0.1, 0.3, 0.5)}[kind]
    if value < th[0]:
        return "ihmal edilebilir"
    return "küçük" if value < th[1] else "orta" if value < th[2] else "büyük"