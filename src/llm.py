"""LLM katmanı: (1) müşterinin Türkçe mesajından tercih çıkarma, (2) öneriyi Türkçe açıklama.

ANTHROPIC_API_KEY tanımlı değilse (veya API hata verirse) kural tabanlı yedek çalışır;
böylece bot anahtarsız da denenebilir. Önemli: LLM ürün SEÇMEZ. Ürünleri istatistik/veri
madenciliği modelleri seçer, LLM yalnızca dili anlar ve sonucu anlatır (halüsinasyonu azaltır).
"""
from __future__ import annotations

import json
import os
import re

import config
from src.recommender import Preferences, RecommendationResult

DEFAULT_MODEL = "claude-sonnet-5-5"

# ---------------------------------------------------------------- kural tabanlı yedek ayrıştırıcı
CATEGORY_TR_KEYWORDS = {
    "SHIRTS": ["gömlek"], "TSHIRTS": ["tişört", "tisort", "t-shirt", "polo"], "UNDERWEAR": ["boxer", "iç çamaşır"], "WAISTCOATS": ["yelek"], "SUITS": ["takım elbise", "takım"], "JACKETS": ["ceket", "mont", "blazer"], "TROUSERS": ["pantolon"],
    "BELTS": ["kemer"], "SWEATERS": ["kazak", "sweatshirt", "hırka"], "SOCKS": ["çorap"],
    "SHORTS": ["şort"], "WALLETS": ["cüzdan"], "JOGGERS": ["eşofman", "jogger"],
    "DENIM": ["kot", "jean"], "FOOTWEAR": ["ayakkabı", "bot ", "spor ayakkabı"], "JERSEYS": ["forma"],
}
COLOUR_TR_KEYWORDS = {
    "NAVY": ["lacivert"], "INDIGO": ["indigo", "koyu mavi"], "BLACK": ["siyah"],
    "GREY": ["gri", "füme"], "CHARCOAL": ["antrasit"], "RED": ["kırmızı", "bordo"],
    "KHAKI": ["haki"], "TAN": ["bej", "camel", "taba", "krem"], "WHITE": ["beyaz"],
    "GREEN": ["yeşil"], "BLUE": ["mavi"], "BROWN": ["kahverengi", "kahve"],
}


def tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def rule_based_preferences(text: str) -> Preferences:
    t = tr_lower(text)
    cats = [c for c, kws in CATEGORY_TR_KEYWORDS.items() if any(k in t for k in kws)]
    colours = [c for c, kws in COLOUR_TR_KEYWORDS.items() if any(k in t for k in kws)]
    size = None
    m = re.search(r"beden\s*[:=]?\s*(xxl|2xl|xl|xs|s|m|l|\d{2})\b", t) or re.search(r"\b(xxl|2xl|xl|xs|s|m|l|\d{2})\s*beden", t)
    if m:
        size = m.group(1).upper().replace("2XL", "XXL")
    budget = None
    if re.search(r"bütçe|altı|altında|kadar|en fazla|maksimum|max", t):
        nums = [int(n.replace(".", "")) for n in re.findall(r"\d[\d\.]*", t) if n.replace(".", "").isdigit()]
        nums = [n for n in nums if n >= 50]
        budget = float(max(nums)) if nums else None
    return Preferences(categories=cats, colours=colours, size=size, max_price=budget)


def _sanitize(d: dict) -> Preferences:
    cats = [str(c).upper() for c in (d.get("categories") or []) if str(c).upper() in config.KNOWN_CATEGORIES]
    colours = [str(c).upper() for c in (d.get("colours") or []) if str(c).strip()]
    size = str(d["size"]).upper() if d.get("size") else None
    if size not in config.VALID_SIZES and not (size and size.isdigit() and 24 <= int(size) <= 52):
        size = None
    try:
        price = float(d["max_price"]) if d.get("max_price") else None
    except (TypeError, ValueError):
        price = None
    return Preferences(cats, colours, size, price)


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.last_error: str | None = None
        self._client = None
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                self.last_error = "anthropic paketi kurulu değil (pip install anthropic)"

    @property
    def available(self) -> bool:
        return self._client is not None

    def _ask(self, system: str, user: str, max_tokens: int = 500) -> str:
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    # ---------------------------------------------------------------- 1) tercih çıkarma
    def extract_preferences(self, text: str, known_colours: list[str]) -> Preferences:
        if self.available:
            system = (
                "Bir erkek giyim mağazasının asistanısın. Müşterinin mesajından tercihleri çıkar ve YALNIZCA "
                'şu JSON\'u döndür: {"categories": [], "colours": [], "size": null, "max_price": null}\n'
                f"categories yalnızca şunlardan seçilebilir: {config.KNOWN_CATEGORIES}.\n"
                f"colours, şu katalog renklerinden en uygun olanlar (İngilizce, büyük harf): {known_colours}.\n"
                "size: XS,S,M,L,XL,XXL'den biri veya null. max_price: sayı veya null. "
                "Belirtilmeyen alanı boş liste/null bırak; tahmin yürütme."
            )
            try:
                raw = self._ask(system, text, max_tokens=300)
                m = re.search(r"\{.*\}", raw, re.S)
                prefs = _sanitize(json.loads(m.group(0)))
                self.last_error = None
                return prefs
            except Exception as e:  # ağ, kota, JSON hatası vb. -> yedeğe geç
                self.last_error = f"{type(e).__name__}: {e}"
        return rule_based_preferences(text)

    # ---------------------------------------------------------------- 2) açıklama
    def explain(self, user_text: str, result: RecommendationResult) -> str:
        facts = {
            "musteri_mesaji": user_text,
            "segment": result.segment_name,
            "notlar": result.notes,
            "oneriler": [r.to_dict() for r in result.recs],
        }
        if self.available and result.recs:
            system = (
                "Bir erkek giyim mağazasının satış danışmanısın. Türkçe, samimi ve profesyonel yaz; en fazla 5 cümle. "
                "YALNIZCA verilen önerileri ve nedenleri kullan: yeni ürün, fiyat, yüzde veya özellik uydurma. "
                "Notlarda kriter gevşetildiyse bunu müşteriye açıkça söyle."
            )
            try:
                out = self._ask(system, json.dumps(facts, ensure_ascii=False), max_tokens=400).strip()
                self.last_error = None
                return out
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
        return self._template_explanation(result)

    @staticmethod
    def _template_explanation(result: RecommendationResult) -> str:
        if not result.recs:
            return "Üzgünüm, bu kriterlere uygun bir ürün bulamadım. Farklı bir renk veya bütçe deneyebilir misiniz?"
        parts = ["Size şu ürünleri öneriyorum; nedenleri yukarıda her ürünün altında yazıyor."]
        if result.notes:
            parts.append("Not: " + " ".join(result.notes))
        return " ".join(parts)