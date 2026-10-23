"""Segment görselleri: PCA dağılımı, kategori payı ısı haritası, segment büyüklükleri."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

import config  # noqa: E402
from src.segmentation import SegmentationResult, _design_matrix  # noqa: E402

# Renk körlüğüne dayanıklı (Okabe-Ito) palet
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]


def _short(seg: int, seg_res: SegmentationResult) -> str:
    n = int(seg_res.profiles.loc[seg, "musteri_sayisi"])
    return f"{seg_res.names[seg].split(',')[0]} (n={n:,})"


def pca_scatter(features: pd.DataFrame, seg: SegmentationResult, path: Path, max_points: int = 20000, seed: int = config.RANDOM_STATE) -> float:
    """Müşterileri 2 boyuta indirip segmentlere göre boyar. Dönen değer: iki bileşenin açıkladığı varyans oranı."""
    X = np.asarray(_design_matrix(features))
    labels = seg.labels.reindex(features.index).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(max_points, len(X)), replace=False)
    pca = PCA(n_components=2, random_state=seed).fit(X)
    Z = pca.transform(X[idx])
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, s in enumerate(sorted(seg.names)):
        m = labels[idx] == s
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.35, color=PALETTE[i % len(PALETTE)], label=_short(s, seg))
    expl = float(pca.explained_variance_ratio_.sum())
    ax.set_xlabel("Birinci bileşen"); ax.set_ylabel("İkinci bileşen")
    ax.set_title(f"Müşteri segmentleri (PCA; iki bileşen varyansın %{expl * 100:.0f}'ini açıklıyor)")
    ax.legend(markerscale=3, fontsize=8, loc="best"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)
    return expl


def category_heatmap(seg: SegmentationResult, path: Path) -> None:
    """Satır = segment, sütun = kategori; hücre = o segmentin harcamasının kategoriye giden yüzdesi."""
    c = seg.share_centroids.loc[sorted(seg.names)]
    c = c.loc[:, c.max(axis=0).sort_values(ascending=False).index] * 100
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(c.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(c.shape[1])); ax.set_xticklabels([config.CATEGORY_TR.get(x, x.lower()) for x in c.columns], rotation=45, ha="right")
    ax.set_yticks(range(c.shape[0])); ax.set_yticklabels([f"Segment {s}" for s in c.index])
    top = c.to_numpy().max()
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            v = c.iat[i, j]
            if v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > top * 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Harcama payı (%)")
    ax.set_title("Segmentlerin kategori harcama dağılımı"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def segment_sizes(seg: SegmentationResult, path: Path) -> None:
    sizes = seg.profiles["musteri_sayisi"].sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"Segment {s}" for s in sizes.index], sizes.values, color=PALETTE[0])
    ax.set_yscale("log"); ax.set_ylabel("Müşteri sayısı (logaritmik ölçek)")
    for i, v in enumerate(sizes.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Segmentlerin büyüklüğü"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def all_plots(features: pd.DataFrame, seg: SegmentationResult, out_dir: Path) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    expl = pca_scatter(features, seg, out_dir / "segment_pca.png")
    category_heatmap(seg, out_dir / "segment_kategori_isi_haritasi.png")
    segment_sizes(seg, out_dir / "segment_buyuklukleri.png")
    return {"pca_aciklanan_varyans": expl}
