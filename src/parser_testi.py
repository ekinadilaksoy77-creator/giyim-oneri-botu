"""Kural tabanlı Türkçe ayrıştırıcının doğruluğunu ölçer.

LLM'i gerçek anahtarla deneyemediğimiz için bot anahtarsız modda bu ayrıştırıcıyı kullanıyor.
Burada elle yazdığımız cümlelerde ayrıştırıcının bulduğu kategori / renk / beden / bütçe, bizim yazdığımız
doğru cevapla karşılaştırılır. Her alan için doğruluk (kaç cümlede tam doğru) hesaplanır.

İki grup cümle var: ilk 25 "normal" cümle; son 10 "zor" cümle (Türkçe karakter yazmama, ek almış kelime,
"numara", "bin" gibi yazımlar). Zor cümleler ayrıştırıcının sınırlarını görmek için eklendi; bulunan hatalar
raporda olduğu gibi yazılır, ayrıştırıcıyı bu cümlelere göre ayarlamadık.
"""
from __future__ import annotations

from src.llm import rule_based_preferences

# (cümle, doğru kategoriler, doğru renkler, doğru beden, doğru bütçe)
CASES = [
    ("İş görüşmesi için lacivert bir gömlek arıyorum, M beden, bütçem 1500", ["SHIRTS"], ["NAVY"], "M", 1500),
    ("siyah kemer lazım", ["BELTS"], ["BLACK"], None, None),
    ("L beden beyaz tişört istiyorum", ["TSHIRTS"], ["WHITE"], "L", None),
    ("1000 liranın altında bir pantolon", ["TROUSERS"], [], None, 1000),
    ("gri kazak bakıyorum, XL beden", ["SWEATERS"], ["GREY"], "XL", None),
    ("yazlık bir şort arıyorum", ["SHORTS"], [], None, None),
    ("kot pantolon, 32 beden, bütçem 2000", ["TROUSERS", "DENIM"], [], "32", 2000),
    ("kahverengi cüzdan", ["WALLETS"], ["BROWN"], None, None),
    ("kırmızı ceket olsun, M beden", ["JACKETS"], ["RED"], "M", None),
    ("çorap ve boxer almak istiyorum", ["UNDERWEAR", "SOCKS"], [], None, None),
    ("düğün için takım elbise, bütçem 5000", ["SUITS"], [], None, 5000),
    ("antrasit yelek arıyorum", ["WAISTCOATS"], ["CHARCOAL"], None, None),
    ("haki eşofman, L beden", ["JOGGERS"], ["KHAKI"], "L", None),
    ("mavi gömlek ve lacivert ceket", ["SHIRTS", "JACKETS"], ["BLUE", "NAVY"], None, None),
    ("S beden siyah tişört, en fazla 600", ["TSHIRTS"], ["BLACK"], "S", 600),
    ("spor ayakkabı bakıyorum, 42 beden", ["FOOTWEAR"], [], "42", None),
    ("yeşil kazak, bütçem 1200", ["SWEATERS"], ["GREEN"], None, 1200),
    ("beden M olan bej pantolon", ["TROUSERS"], ["TAN"], "M", None),
    ("hırka arıyorum", ["SWEATERS"], [], None, None),
    ("2.500 TL bütçeyle mont", ["JACKETS"], [], None, 2500),
    ("sadece bir şey bakıyorum", [], [], None, None),
    ("beyaz gömlek 3 adet", ["SHIRTS"], ["WHITE"], None, None),
    ("indigo kot lazım, bütçem 1800", ["DENIM"], ["INDIGO"], None, 1800),
    ("XXL beden siyah pantolon", ["TROUSERS"], ["BLACK"], "XXL", None),
    ("bordo kazak istiyorum", ["SWEATERS"], ["RED"], None, None),
    # ---- zor cümleler ----
    ("lacivert gomlek ariyorum", ["SHIRTS"], ["NAVY"], None, None),
    ("beyaz tisort lazim, M beden", ["TSHIRTS"], ["WHITE"], "M", None),
    ("gömleği siyah olarak görmek istiyorum", ["SHIRTS"], ["BLACK"], None, None),
    ("ayakkabı numaram 42", ["FOOTWEAR"], [], "42", None),
    ("bin beş yüz liraya kadar pantolon", ["TROUSERS"], [], None, 1500),
    ("1500 tl'ye kadar bir ceket bakıyorum", ["JACKETS"], [], None, 1500),
    ("kırmızı olmasın, gri kazak", ["SWEATERS"], ["GREY"], None, None),
    ("kazak ya da hırka", ["SWEATERS"], [], None, None),
    ("kış için sıcak tutan bir şey", [], [], None, None),
    ("büyük beden siyah tişört", ["TSHIRTS"], ["BLACK"], None, None),
]


def run() -> dict:
    fields = ("kategori", "renk", "beden", "butce")
    ok = {f: 0 for f in fields}
    errors = []
    for text, cats, cols, size, budget in CASES:
        p = rule_based_preferences(text)
        got = {"kategori": sorted(p.categories), "renk": sorted(p.colours), "beden": p.size,
               "butce": int(p.max_price) if p.max_price else None}
        want = {"kategori": sorted(cats), "renk": sorted(cols), "beden": size, "butce": budget}
        for f in fields:
            if got[f] == want[f]:
                ok[f] += 1
            else:
                errors.append({"cumle": text, "alan": f, "beklenen": want[f], "bulunan": got[f]})
    n = len(CASES)
    all_ok = n - len({e["cumle"] for e in errors})
    return {"cumle_sayisi": n, "dogruluk": {f: round(ok[f] / n, 3) for f in fields},
            "tum_alanlar_dogru": round(all_ok / n, 3), "hatalar": errors}
