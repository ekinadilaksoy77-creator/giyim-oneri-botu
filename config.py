"""Proje genelinde kullanılan ayarlar ve sabitler."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42

# Veri setindeki ürün kategorileri (Kaggle "Clothes Retail Store" veri seti).
KNOWN_CATEGORIES = [
    "SHIRTS", "TSHIRTS", "SUITS", "UNDERWEAR", "WAISTCOATS", "JERSEYS", "JACKETS", "TROUSERS", "BELTS", "SWEATERS",
    "SOCKS", "SHORTS", "WALLETS", "JOGGERS", "DENIM", "FOOTWEAR", "MASKS",
]
# Öneri sisteminde kullanmayacağımız kategoriler.
EXCLUDED_CATEGORIES = {"MASKS"}

# Müşteriye gösterirken kullanacağımız Türkçe karşılıklar.
CATEGORY_TR = {
    "SHIRTS": "gömlek",
    "TSHIRTS": "tişört",
    "UNDERWEAR": "iç çamaşırı (boxer)",
    "WAISTCOATS": "yelek",
    "SUITS": "takım elbise",
    "JERSEYS": "forma",
    "JACKETS": "ceket",
    "TROUSERS": "pantolon",
    "BELTS": "kemer",
    "SWEATERS": "kazak",
    "SOCKS": "çorap",
    "SHORTS": "şort",
    "WALLETS": "cüzdan",
    "JOGGERS": "eşofman altı",
    "DENIM": "kot",
    "FOOTWEAR": "ayakkabı",
    "MASKS": "maske",
}

VALID_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
NOT_CAPTURED = "NOT CAPTURED"

# Öneri ayarları
DEFAULT_TOP_K = 5
MIN_BUYERS_PER_ITEM = 3     # Çok az kişinin aldığı ürünleri kataloğa almayız (gürültü)
BUDGET_TOLERANCE = 0.25    # İstenen renkteki ürün bütçeyi en fazla %25 aşabilir (uyarıyla gösterilir)
CF_WEIGHT = 0.6             # Hibrit skorda "benzer ürün" (collaborative filtering) ağırlığı

# Ürün kodu ön eki -> kategori. Kodlu ürünlerde ("MSH ITMSH03378 Chiseled LS Green XL") kategori adı geçmediği için.
# Güven düzeyi, örnek satırlara bakılarak verildi. LS/SS/SL = kol tipi, FF = düz kesim (pantolon), 5P = 5 cep (kot).
#   KANITLI : MSH gömlek, MTR pantolon (bel bedeni 32/34), MDE kot ("5P Dark Wash"), MBL kemer ("MENS BELT" ile geliyor)
#   GÜÇLÜ   : MJA ceket (Bomber/Biker), MWL ve WLT cüzdan (Bifold/Passcase), MBX boxer, MFW ayakkabı ("MENS FOOTWEAR"),
#             MWC yelek (Vest/Bandi), MSX çorap ("Full length ... Foot")
#   VARSAYIM (raporda "tahmini eşleme" diye belirt): MTS ve MRG tişört, MSS gömlek, MSW kazak/sweatshirt, MSO şort
# B ile başlayanlar (BSH, BTR, BDE, BSO...) bilerek YOK: bedenleri farklı (ES/EL/EEL), ne oldukları belli değil.
# Çanta/ambalaj/promosyon kodları (CBG, TBG, LBG, BBG, CTrims, WOthers, HHP, FFB, FFT, Promo, CARRY BAG) de YOK.
PREFIX_CATEGORY = {
    "MSH": "SHIRTS", "MTR": "TROUSERS", "MDE": "DENIM", "MBL": "BELTS",
    "MJA": "JACKETS", "MWL": "WALLETS", "WLT": "WALLETS", "MBX": "UNDERWEAR",
    "MFW": "FOOTWEAR", "MWC": "WAISTCOATS", "MSX": "SOCKS",
    "MTS": "TSHIRTS", "MRG": "TSHIRTS", "MSS": "SHIRTS", "MSW": "SWEATERS", "MSO": "SHORTS",
}