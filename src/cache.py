"""Eğitilmiş modeli diske kaydeder: `chat` her açılışta beş yıllık veriyi yeniden okuyup eğitmesin."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import config
from src.pipeline import Artifacts

CACHE_VERSION = "2"        # model kodu değişirse bunu artır, eski önbellek geçersiz olur


def cache_path() -> Path:
    return Path(config.PROCESSED_DIR) / "model.pkl"


def signature(paths: list[Path]) -> str:
    """Veri dosyaları veya ürün-kod eşlemesi değişince önbellek otomatik geçersiz olur."""
    h = hashlib.md5()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    h.update(json.dumps(config.PREFIX_CATEGORY, sort_keys=True).encode())
    h.update(CACHE_VERSION.encode())
    return h.hexdigest()


def load(sig: str) -> Artifacts | None:
    path = cache_path()
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj["art"] if obj.get("sig") == sig else None
    except Exception:
        return None


def save(sig: str, art: Artifacts) -> None:
    """Büyük tabloları (df, features) kaydetmeyiz; bot yalnızca öneri motoruna ihtiyaç duyar."""
    light = replace(art, df=None, features=None)
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"sig": sig, "art": light}, f, protocol=pickle.HIGHEST_PROTOCOL)