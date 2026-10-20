import pandas as pd

import config
from src import data_loader


def test_clean_drops_invalid_amounts_and_fills_missing(raw_df):
    df, report = data_loader.clean(raw_df)
    assert (df["amount"] >= 1).all()                       # TC: geçersiz tutarlar atılmış
    assert df["colour"].notna().all() and df["size"].notna().all()
    assert config.NOT_CAPTURED in set(df["colour"])        # eksik renkler doldurulmuş
    assert report["temiz_satir"] == len(df) < report["ham_satir"]


def test_category_derived_from_description_when_missing():
    inv = pd.Series([None, "SHIRTS", None, "weird text"])
    desc = pd.Series(["MENS NAVY sweatshirt", "x", "Men Jogger Black", "unknown thing"])
    out = data_loader.categorize(inv, desc)
    assert out.iloc[0] == "SWEATERS"      # 'sweatshirt' içinde 'shirt' var ama SWEATERS önce gelmeli
    assert out.iloc[1] == "SHIRTS"
    assert out.iloc[2] == "JOGGERS"
    assert pd.isna(out.iloc[3])


def test_all_categories_known(clean_df):
    assert set(clean_df["category"]) <= set(config.KNOWN_CATEGORIES)
    assert not (set(clean_df["category"]) & config.EXCLUDED_CATEGORIES)


def test_missing_columns_raise_clear_error(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    try:
        data_loader.read_raw(p)
    except ValueError as e:
        assert "sütunlar yok" in str(e)
    else:
        raise AssertionError("ValueError bekleniyordu")


def test_multiple_year_files_are_combined(tmp_path, raw_df):
    a, b = raw_df.iloc[:500], raw_df.iloc[500:1000]
    a.to_csv(tmp_path / "y2021.csv", index=False)
    b.to_csv(tmp_path / "y2022.csv", index=False)
    df, report = data_loader.load_clean([tmp_path / "y2021.csv", tmp_path / "y2022.csv"])
    assert "y2021.csv" in report["dosya"] and "y2022.csv" in report["dosya"]
    assert report["ham_satir"] == 1000


def test_real_format_category_text_is_mapped():
    # Gerçek veriden bir satır biçimi: inventory_category = "MENS BELT", description kodlu
    out = data_loader.categorize(pd.Series(["MENS BELT", "MENS SUIT"]), pd.Series(["MBL ITA16BLT004 Regular Casual Tan 32", "x"]))
    assert list(out) == ["BELTS", "SUITS"]


def test_coded_products_mapped_by_prefix():
    inv = pd.Series(["MSH ITMSH03378 Chiseled LS Green XL", None, "MDE ITMDN00346 Brooklyn 5P Dark Wash 34",
                     "CARRY BAG", "CBG ITCBR00001 NA NA White 12x4x16``"])
    desc = pd.Series(["MSH ITMSH03378 Chiseled LS Green XL", "MTR ITMTR00484 Brooklyn FF Grey 32",
                      "MDE ITMDN00346 Brooklyn 5P Dark Wash 34", "CBG ITCBR00001 NA NA KHAKI 12X4X16", "x"])
    out = data_loader.categorize(inv, desc)
    assert list(out.iloc[:3]) == ["SHIRTS", "TROUSERS", "DENIM"]
    assert out.iloc[3:].isna().all()          # çantalar ürün değil, kategorisiz kalmalı


def test_colour_and_size_filled_from_description():
    df = pd.DataFrame({
        "description": ["MSH ITMSH03378 Chiseled LS Green XL", "MDE ITMDN00346 Brooklyn 5P Dark Wash 34",
                        "MTS ITMTS00990 Regular SS White 2XL", "MSH ITMSH01078 Slim LS Red M"],
        "colour": ["NOT-CAPTURED", None, "NOT-CAPTURED", "Red"],
        "size": ["NOT-CAPTURED", None, "NOT-CAPTURED", "M"],
    })
    out = data_loader.fill_from_description(df)
    assert list(out["colour"]) == ["Green", "Dark Wash", "White", "Red"]
    assert list(out["size"]) == ["XL", "34", "XXL", "M"]


def test_more_prefixes_and_keywords():
    inv = pd.Series(["MJA ITMJA00170 Bomber LS GunMetal M", "MBX ITMBX00065 Regular Boxer Grey XL",
                     "MWL ITMWL00013 Regular Bifold Black 24CM", "MENS FOOTWEAR", "Solid Pique PoloSizeXL",
                     "BSH ITBSH00914 Regular LS Navy EEL", "TBG ITPTB00004 NA NA Brown 18``x9``x10``"])
    desc = pd.Series(["a", "b", "c", "MFW ITA16FTW001 Regular Chukka Tan-Crust 9", "Solid Pique PoloSizeXL", "d", "e"])
    out = data_loader.categorize(inv, desc)
    got = out.astype(object).where(out.notna(), "YOK").tolist()   # hata olursa hangi değer boş görünsün
    assert got[:5] == ["JACKETS", "UNDERWEAR", "WALLETS", "FOOTWEAR", "TSHIRTS"], got
    assert got[5:] == ["YOK", "YOK"], got       # B'li kodlar ve çantalar bilerek kategorisiz