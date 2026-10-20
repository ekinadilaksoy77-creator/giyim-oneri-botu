"""Müşteri-ürün eşleştirme botu (komut satırı).

Kullanım örnekleri (bota yazılacak mesajlar):
  "İş için lacivert bir gömlek arıyorum, M beden, bütçem 1500"   -> yeni müşteri önerisi
  "müşteri U00012"                                                 -> mevcut müşteri önerisi
  "paket gömlek"                                                   -> gömlekle birlikte alınanlar
  "çık"                                                            -> çıkış
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config
from src.llm import LLMClient, tr_lower
from src.pipeline import Artifacts
from src.recommender import RecommendationResult, UnknownCustomerError

HELP = (
    "Ne aradığınızı yazın (örn. 'lacivert gömlek, M beden, bütçem 1500'), "
    "kayıtlı müşteri için 'müşteri <id>', birlikte alınanlar için 'paket gömlek'. Çıkmak için 'çık' (veya 'q', 'exit', Ctrl+C)."
)


EXIT_WORDS = {"çık", "cik", "çıkış", "cikis", "q", "exit", "quit", "/exit", "/quit", "bye", "görüşürüz"}


def is_exit(message: str) -> bool:
    m = message.strip()
    return tr_lower(m) in EXIT_WORDS or m.lower() in EXIT_WORDS   # 'EXIT' ve 'ÇIK' ikisi de çalışsın


def format_result(result: RecommendationResult) -> str:
    if not result.recs:
        return "Uygun ürün bulunamadı."
    lines = []
    if result.segment_name:
        lines.append(f"Tahmini müşteri grubu: {result.segment_name}")
    for i, r in enumerate(result.recs, 1):
        cat = config.CATEGORY_TR.get(r.category, r.category)
        lines.append(f"{i}. {cat.capitalize()} - {r.colour}  (tipik fiyat ~{r.price:.0f})")
        lines += [f"     • {why}" for why in r.reasons]
    return "\n".join(lines)


@dataclass
class Response:
    """Botun cevabı: ya düz bir mesaj (text) ya da öneri listesi (result) + açıklama."""
    text: str = ""
    result: RecommendationResult | None = None
    explanation: str = ""
    title: str = ""


class StyleBot:
    def __init__(self, artifacts: Artifacts, llm: LLMClient | None = None):
        self.art = artifacts
        self.llm = llm or LLMClient()
        rec = artifacts.recommender
        self.known_colours = sorted({c for c in rec.items["colour"] if c != config.NOT_CAPTURED})

    def reply(self, message: str) -> str:
        r = self.respond(message)
        if r.result is None:
            return r.text
        return format_result(r.result) + ("\n\n" + r.explanation if r.explanation else "")

    def respond(self, message: str) -> Response:
        msg = message.strip()
        low = tr_lower(msg)
        rec = self.art.recommender

        m = re.match(r"^(?:müşteri|musteri)\s+(\S+)", low)
        if m:
            uid = msg.split()[1]
            try:
                result = rec.recommend_existing(uid)
            except UnknownCustomerError:
                return Response(text=f"'{uid}' numaralı müşteri kayıtlarda yok. Yeni müşteri gibi devam etmek için ne aradığınızı yazın.")
            return Response(result=result, title=f"Müşteri {uid} için öneriler",
                            explanation=self.llm.explain(f"{uid} numaralı mevcut müşteri için öneri", result))

        m = re.match(r"^paket\s+(.+)$", low)
        if m:
            prefs = self.llm.extract_preferences(m.group(1), self.known_colours)
            if not prefs.categories:
                return Response(text="Hangi ürünle birlikte alınanları görmek istersiniz? Örn: 'paket gömlek'.")
            recs = rec.bundle_for(prefs.categories[0])
            if not recs:
                return Response(text="Bu ürün için yeterince güçlü bir birliktelik kuralı bulunamadı. "
                                     "Gömlek gibi çok satan ürünler her sepette olduğu için kural çıkmaz; 'paket kazak' veya 'paket şort' deneyin.")
            return Response(result=RecommendationResult(recs), title="Birlikte en çok alınanlar")

        prefs = self.llm.extract_preferences(msg, self.known_colours)
        if prefs.is_empty():
            return Response(text="Ne aradığınızı tam anlayamadım. " + HELP)
        result = rec.recommend_new(prefs)
        return Response(result=result, title="Size önerilenler", explanation=self.llm.explain(msg, result))

    def run(self) -> None:
        try:
            from src import ui
        except ImportError:        # rich kurulu değilse düz metin arayüzü
            return self._run_plain()
        ui.run(self)

    def _run_plain(self) -> None:
        mode = f"LLM: {self.llm.model}" if self.llm.available else "LLM yok (ANTHROPIC_API_KEY tanımsız) - kural tabanlı mod"
        print(f"Giyim öneri botu hazır. [{mode}]\n{HELP}\n")
        while True:
            try:
                msg = input("Siz: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not msg:
                continue
            if is_exit(msg):
                break
            print("\nBot:\n" + self.reply(msg) + "\n")
            if self.llm.last_error:
                print(f"(LLM uyarısı, yedek mod kullanıldı: {self.llm.last_error})\n")