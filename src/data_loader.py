"""Veri yükleme ve temizleme.

Kaggle "Clothes Retail Store" (nishchay331/retail-store) veri setinin sütunları:
user_id, bill_id, line_item_amount, bill_discount, transaction_date,
description, inventory_category, colour, size, zone_name, store_name, year

Her satır = bir fatura (bill) içindeki TEK bir ürün kalemidir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config

REQUIRED_COLUMNS = [
    "user_id", "bill_id", "line_item_amount", "bill_discount",
    "transaction_date", "description", "inventory_category",
    "colour", "size", "zone_name", "store_name",
]

# Sıra ÖNEMLİ: ilk eşleşen kategori seçilir. ("sweatshirt" içinde "shirt" geçtiği için
# SWEATERS, SHIRTS'ten önce kontrol edilmelidir.)
CATEGORY_KEYWORDS = [
    ("SWEATERS", ["sweater", "sweat", "hoodie", "pullover", "cardigan"]),
    ("JERSEYS", ["jersey"]),
    ("JOGGERS", ["jogger", "track pant", "trackpant"]),
    ("SHORTS", ["shorts"]),
    ("DENIM", ["denim", "jeans"]),
    ("SUITS", ["suit"]),
    ("JACKETS", ["jacket", "blazer", "coat"]),
    ("TROUSERS", ["trouser", "chino"]),
    ("BELTS", ["belt"]),
    ("SOCKS", ["sock"]),
    ("WALLETS", ["wallet"]),
    ("FOOTWEAR", ["footwear", "shoe", "boot", "sneaker", "sandal", "loafer"]),
    ("UNDERWEAR", ["boxer"]),
    ("WAISTCOATS", ["waistcoat"]),
    ("MASKS", ["mask"]),
    ("TSHIRTS", ["t-shirt", "tshirt", "t shirt", "polo", "henley"]),   # SHIRTS'ten ÖNCE
    ("SHIRTS", ["shirt"]),
]


def find_data_files(raw_dir: Path | None = None) -> list[Path]:
    """data/raw içindeki CSV'leri bulur (birden fazla yıl olabilir). Gerçek veri varsa onları, yoksa sentetiği seçer."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"'{raw_dir}' klasöründe CSV bulunamadı.\n"
            "Gerçek veri için README'deki Kaggle indirme adımını uygula,\n"
            "ya da deneme için sentetik veri üret:  python main.py generate-data"
        )
    real = [p for p in csvs if not p.name.startswith("sentetik")]
    return real or csvs


def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user_id": str, "bill_id": str}, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV'de beklenen sütunlar yok: {missing}\nBulunan sütunlar: {list(df.columns)}"
        )
    return df


def categorize(inventory_category: pd.Series, description: pd.Series) -> pd.Series:
    """Kategori sütunu düzgünse onu kullanır, değilse açıklama (description) metninden türetir."""
    cat = inventory_category.astype("string").str.strip().str.upper()
    known = cat.where(cat.isin(config.KNOWN_CATEGORIES))

    text = (inventory_category.fillna("").astype(str) + " " + description.fillna("").astype(str)).str.lower()
    conditions = [text.str.contains("|".join(kws), regex=True) for _, kws in CATEGORY_KEYWORDS]
    names = [name for name, _ in CATEGORY_KEYWORDS]
    derived = pd.Series(np.select(conditions, names, default=""), index=text.index).replace("", pd.NA)

    # Kodlu ürünler: ilk kelime ön ektir (inventory_category boşsa description'a bakılır)
    first = inventory_category.where(inventory_category.notna(), description).astype("string").str.split().str[0].str.upper()
    by_prefix = first.map(config.PREFIX_CATEGORY)

    return known.fillna(derived).fillna(by_prefix)


_SIZE_RE = r"(XXS|XS|S|M|L|XL|XXL|2XL|3XL|4XL|\d{2})"


def fill_from_description(df: pd.DataFrame) -> pd.DataFrame:
    """colour/size sütunu boşsa ("NOT-CAPTURED"), açıklamadan çıkarır.

    Örn. "MSH ITMSH03378 Chiseled LS Green XL" -> renk: Green, beden: XL
         "MDE ITMDN00346 Brooklyn 5P Dark Wash 34" -> renk: Dark Wash, beden: 34
    """
    desc = df["description"].fillna("").astype(str).str.strip()
    size_new = desc.str.extract(r"\s" + _SIZE_RE + r"\s*$", flags=2)[0].str.upper()
    size_new = size_new.where(size_new != "2XL", "XXL")
    colour_new = desc.str.extract(r"\s(?:LS|SS|SL|FF|5P|HS)\s+(.+?)\s+" + _SIZE_RE + r"\s*$", flags=2)[0]

    no_size = df["size"].isna() | df["size"].astype(str).str.upper().str.replace("-", " ").isin(["", "NOT CAPTURED", "NAN"])
    no_col = df["colour"].isna() | df["colour"].astype(str).str.upper().str.replace("-", " ").isin(["", "NOT CAPTURED", "NAN"])
    df = df.copy()
    df.loc[no_size & size_new.notna(), "size"] = size_new
    df.loc[no_col & colour_new.notna(), "colour"] = colour_new
    return df


def _normalize_label(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip().str.upper().str.replace("-", " ", regex=False)
    s = s.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    return s.fillna(config.NOT_CAPTURED)


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ham veriyi temizler. Ne kadar satır attığımızı raporlar (rapora yazmak için)."""
    report = {"ham_satir": int(len(raw))}
    df = raw.copy()

    df = df.dropna(subset=["user_id", "bill_id", "transaction_date"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date"])
    report["kimlik_veya_tarih_eksik_sonrasi"] = int(len(df))

    df["amount"] = pd.to_numeric(df["line_item_amount"], errors="coerce").round(2)
    df = df[df["amount"] >= 1]
    report["tutar_gecerli_sonrasi"] = int(len(df))

    df["category"] = categorize(df["inventory_category"], df["description"])
    unmapped_mask = df["category"].isna()
    raw_cat = df.loc[unmapped_mask, "inventory_category"].fillna("(boş)").astype(str)
    report["kategorisi_belirlenemeyen_toplam_satir"] = int(unmapped_mask.sum())
    report["kategorisi_belirlenemeyen_en_cok_gecenler"] = {
        str(k): int(v) for k, v in raw_cat.value_counts().head(15).items()}
    # Ürün kodları satır satır farklı olduğu için ilk iki kelimeye göre gruplamak daha anlamlı
    first_words = raw_cat.str.split().str[:2].str.join(" ")
    report["belirlenemeyen_ilk_iki_kelimeye_gore"] = {
        str(k): int(v) for k, v in first_words.value_counts().head(40).items()}
    df = df.dropna(subset=["category"])
    report["maske_satiri_atilan"] = int((df["category"] == "MASKS").sum())
    df = df[~df["category"].isin(config.EXCLUDED_CATEGORIES)]
    report["kategori_belirlenen_sonrasi"] = int(len(df))

    n_before = int((df["colour"].astype(str).str.upper().str.replace("-", " ") == "NOT CAPTURED").sum())
    df = fill_from_description(df)
    df["colour"] = _normalize_label(df["colour"])
    df["size"] = _normalize_label(df["size"]).replace({"2XL": "XXL"})
    report["renk_bos_once"] = n_before
    report["renk_bos_sonra"] = int((df["colour"] == config.NOT_CAPTURED).sum())
    report["beden_bos_sonra"] = int((df["size"] == config.NOT_CAPTURED).sum())
    df["bill_discount"] = pd.to_numeric(df["bill_discount"], errors="coerce").fillna(0).round(0)
    df["item_id"] = df["category"].astype(str) + " | " + df["colour"].astype(str)

    cols = [
        "user_id", "bill_id", "transaction_date", "amount", "bill_discount",
        "description", "category", "colour", "size", "zone_name", "store_name", "item_id",
    ]
    df = df[cols].reset_index(drop=True)
    df["category"] = df["category"].astype(str)
    report["temiz_satir"] = int(len(df))
    report["musteri_sayisi"] = int(df["user_id"].nunique())
    report["fatura_sayisi"] = int(df["bill_id"].nunique())
    report["tarih_araligi"] = [str(df["transaction_date"].min().date()), str(df["transaction_date"].max().date())]
    return df, report


def load_clean(path: Path | list[Path] | None = None) -> tuple[pd.DataFrame, dict]:
    """Bir CSV, CSV listesi veya (None ise) data/raw içindeki tüm CSV'ler okunur ve birleştirilir."""
    if path is None:
        paths = find_data_files()
    elif isinstance(path, (list, tuple)):
        paths = [Path(p) for p in path]
    else:
        paths = [Path(path)]
    raw = pd.concat([read_raw(p) for p in paths], ignore_index=True)
    df, report = clean(raw)
    report["dosya"] = ", ".join(p.name for p in paths)
    return df, report