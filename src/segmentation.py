"""Müşteri segmentasyonu: k-means (RFM + kategori payları)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from src.features import SHARE_PREFIX, share_columns


@dataclass
class SegmentationResult:
    k: int
    labels: pd.Series                 # user_id -> segment numarası
    names: dict[int, str]             # segment numarası -> okunur isim
    profiles: pd.DataFrame            # segment özet tablosu (rapor için)
    share_centroids: pd.DataFrame     # segment x kategori: ortalama harcama payı
    silhouette_by_k: dict[int, float]

    def assign_by_categories(self, categories: list[str]) -> int | None:
        """Yeni müşteri: söylediği kategorilere en çok benzeyen segmenti bulur (kosinüs benzerliği)."""
        cats = [c for c in categories if c in self.share_centroids.columns]
        if not cats:
            return None
        vec = pd.Series(0.0, index=self.share_centroids.columns)
        vec[cats] = 1.0 / len(cats)
        m = self.share_centroids.to_numpy()
        v = vec.to_numpy()
        sims = (m @ v) / (np.linalg.norm(m, axis=1) * np.linalg.norm(v) + 1e-12)
        return int(self.share_centroids.index[int(np.argmax(sims))])


def _design_matrix(features: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame(index=features.index)
    X["recency_days"] = features["recency_days"]
    X["log_frequency"] = np.log1p(features["frequency"])
    X["log_monetary"] = np.log1p(features["monetary"])
    X = X.join(features[share_columns(features)])
    return StandardScaler().fit_transform(X)


def fit_segments(features: pd.DataFrame, k: int | None = None, k_range=range(3, 7),
                 random_state: int = config.RANDOM_STATE) -> SegmentationResult:
    """k verilmezse silhouette skoruna bakarak k_range içinden en iyisini seçer."""
    X = _design_matrix(features)
    sil: dict[int, float] = {}
    if k is None:
        for kk in k_range:
            labels = KMeans(n_clusters=kk, n_init=5, random_state=random_state).fit_predict(X)
            sil[kk] = float(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=random_state))
        k = max(sil, key=sil.get)

    labels = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit_predict(X)
    labels = pd.Series(labels, index=features.index, name="segment")

    cols = share_columns(features)
    centroids = features[cols].groupby(labels).mean()
    centroids.columns = [c[len(SHARE_PREFIX):] for c in centroids.columns]

    prof = features[["recency_days", "frequency", "monetary"]].groupby(labels).mean().round(1)
    prof.insert(0, "musteri_sayisi", labels.value_counts().sort_index())
    median_monetary = features["monetary"].median()
    names = {}
    for seg in prof.index:
        top = centroids.loc[seg].sort_values(ascending=False).head(2).index
        top_tr = " / ".join(config.CATEGORY_TR.get(c, c.lower()) for c in top)
        level = "yüksek harcamalı" if prof.loc[seg, "monetary"] > median_monetary else "düşük/orta harcamalı"
        names[int(seg)] = f"Segment {seg}: {top_tr} ağırlıklı, {level}"
    prof["isim"] = [names[int(s)] for s in prof.index]
    prof["en_cok_kategori"] = [", ".join(centroids.loc[s].sort_values(ascending=False).head(3).index) for s in prof.index]
    return SegmentationResult(k=k, labels=labels, names=names, profiles=prof,
                              share_centroids=centroids, silhouette_by_k=sil)
