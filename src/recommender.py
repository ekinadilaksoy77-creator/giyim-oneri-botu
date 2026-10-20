"""Hibrit öneri motoru.

İki durum var:
 1) MEVCUT müşteri  -> geçmişi var:  skor = a * (benzer ürünler) + (1-a) * (kendi segmentinde popülerlik)
 2) YENİ müşteri    -> geçmişi yok ("soğuk başlangıç"): söylediği tercihlerden segment tahmin edilir,
                       o segmentte popüler ürünler filtrelenip sıralanır.

"Ürün" = kategori + renk  (örn. "SHIRTS | NAVY"). Beden ayrı bir filtre olarak kullanılır.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.segmentation import SegmentationResult


class UnknownCustomerError(KeyError):
    pass


@dataclass
class Preferences:
    """Yeni müşterinin söylediği tercihler (LLM veya kural tabanlı ayrıştırıcıdan gelir)."""
    categories: list[str] = field(default_factory=list)   # örn. ["SHIRTS"]
    colours: list[str] = field(default_factory=list)      # örn. ["NAVY", "INDIGO"] (alt metin eşleşmesi)
    size: str | None = None                                # örn. "M"
    max_price: float | None = None

    def is_empty(self) -> bool:
        return not (self.categories or self.colours or self.size or self.max_price)


@dataclass
class Recommendation:
    item_id: str
    category: str
    colour: str
    price: float
    score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "urun": f"{config.CATEGORY_TR.get(self.category, self.category)} - {self.colour}",
            "kategori": self.category, "renk": self.colour,
            "tipik_fiyat": round(self.price), "skor": round(self.score, 3), "nedenler": self.reasons,
        }


@dataclass
class RecommendationResult:
    recs: list[Recommendation]
    notes: list[str] = field(default_factory=list)   # örn. "Beden filtresi gevşetildi"
    segment_id: int | None = None
    segment_name: str | None = None


def _norm(v: np.ndarray) -> np.ndarray:
    m = v.max() if len(v) else 0
    return v / m if m > 0 else v


class HybridRecommender:
    def __init__(self, df_train: pd.DataFrame, seg: SegmentationResult, rules: pd.DataFrame,
                 cf_weight: float = config.CF_WEIGHT, min_buyers: int = config.MIN_BUYERS_PER_ITEM):
        self.seg = seg
        self.rules = rules
        self.cf_weight = cf_weight

        # --- Ürün kataloğu ---
        # Rengi bilinmeyen ("NOT CAPTURED") satırlar müşteriye gösterilecek bir ürün değildir: kataloğa girmez.
        known = df_train[df_train["colour"] != config.NOT_CAPTURED]
        if len(known):
            df_train = known
        items = df_train.groupby("item_id").agg(
            category=("category", "first"), colour=("colour", "first"),
            price=("amount", "median"), n_buyers=("user_id", "nunique"),
        )
        sizes = (df_train[df_train["size"] != config.NOT_CAPTURED]
                 .groupby("item_id")["size"].agg(lambda s: set(s)))
        items["sizes"] = sizes.reindex(items.index).apply(lambda x: x if isinstance(x, set) else set())
        keep = items[items["n_buyers"] >= min_buyers]
        self.items = (keep if len(keep) else items).sort_values("n_buyers", ascending=False)
        self.item_ids = list(self.items.index)
        self._item_pos = {i: p for p, i in enumerate(self.item_ids)}

        # --- Müşteri x ürün matrisi (aldı=1 / almadı=0) ---
        d = df_train[df_train["item_id"].isin(self._item_pos)]
        self.user_ids = list(d["user_id"].unique())
        self._user_pos = {u: p for p, u in enumerate(self.user_ids)}
        rows = d["user_id"].map(self._user_pos).to_numpy()
        cols = d["item_id"].map(self._item_pos).to_numpy()
        m = sparse.csr_matrix((np.ones(len(d)), (rows, cols)), shape=(len(self.user_ids), len(self.item_ids)))
        m.data[:] = 1.0
        m.sum_duplicates()
        m.data[:] = 1.0
        self.user_item = m

        # --- Ürün-ürün benzerliği (kosinüs): aynı müşterilerce birlikte alınan ürünler benzerdir ---
        self.item_sim = cosine_similarity(m.T.tocsr())
        np.fill_diagonal(self.item_sim, 0.0)

        # --- Popülerlik: genel ve segment bazlı (o segmentteki müşterilerin yüzde kaçı almış) ---
        self.global_pop = np.asarray(m.sum(axis=0)).ravel() / max(len(self.user_ids), 1)
        user_seg = seg.labels.reindex(self.user_ids)
        self.seg_pop: dict[int, np.ndarray] = {}
        for s in seg.names:
            idx = np.where((user_seg == s).to_numpy())[0]
            self.seg_pop[s] = (np.asarray(m[idx].sum(axis=0)).ravel() / len(idx)) if len(idx) else np.zeros(len(self.item_ids))

    # ------------------------------------------------------------------ yardımcılar
    def _make_rec(self, pos: int, score: float, reasons: list[str]) -> Recommendation:
        row = self.items.iloc[pos]
        return Recommendation(self.item_ids[pos], row["category"], row["colour"], float(row["price"]), float(score), reasons)

    def popular_items(self, k: int = config.DEFAULT_TOP_K, exclude: set[str] | None = None) -> list[str]:
        order = np.argsort(-self.global_pop)
        out = [self.item_ids[p] for p in order if not exclude or self.item_ids[p] not in exclude]
        return out[:k]

    # ------------------------------------------------------------------ mevcut müşteri
    def recommend_existing(self, user_id: str, k: int = config.DEFAULT_TOP_K, cf_weight: float | None = None) -> RecommendationResult:
        if user_id not in self._user_pos:
            raise UnknownCustomerError(user_id)
        a = self.cf_weight if cf_weight is None else cf_weight
        up = self._user_pos[user_id]
        bought = self.user_item[up].indices                       # satın alınan ürün konumları
        seg_id = int(self.seg.labels.get(user_id, -1))

        cf = _norm(np.asarray(self.user_item[up] @ self.item_sim).ravel())
        sp = _norm(self.seg_pop.get(seg_id, self.global_pop))
        score = a * cf + (1 - a) * sp
        score[bought] = -np.inf                                    # zaten aldığı ürünü tekrar önermeyiz

        top = [p for p in np.argsort(-score)[:k] if np.isfinite(score[p])]
        recs = []
        for p in top:
            reasons = []
            if cf[p] > 0 and len(bought):
                best = bought[int(np.argmax(self.item_sim[bought, p]))]
                reasons.append(f"Daha önce aldığınız '{self._label(best)}' ile benzer alışveriş yapan müşteriler bunu da aldı.")
            if sp[p] > 0 and seg_id in self.seg_pop:
                reasons.append(f"Sizinle aynı gruptaki müşterilerde bu ürünü alma oranı: %{self.seg_pop[seg_id][p] * 100:.1f}")
            recs.append(self._make_rec(p, score[p], reasons))
        return RecommendationResult(recs, [], seg_id if seg_id >= 0 else None, self.seg.names.get(seg_id))

    def _label(self, pos: int) -> str:
        row = self.items.iloc[pos]
        return f"{config.CATEGORY_TR.get(row['category'], row['category'])} - {row['colour']}"

    # ------------------------------------------------------------------ yeni müşteri (soğuk başlangıç)
    def recommend_new(self, prefs: Preferences, k: int = config.DEFAULT_TOP_K, use_segment: bool = True) -> RecommendationResult:
        seg_id = self.seg.assign_by_categories(prefs.categories) if (use_segment and prefs.categories) else None
        if seg_id is not None:
            base = _norm(self.seg_pop[seg_id]) + 0.3 * _norm(self.global_pop)
        else:
            base = _norm(self.global_pop)

        cats = self.items["category"].to_numpy()
        colours = self.items["colour"].to_numpy()
        # Zorunlu (filtre) kriterler. Renk filtre DEĞİL, skor artırıcıdır: "ürün" = kategori+renk olduğu için
        # tam renk eşleşmesi çoğu zaman tek üründür; filtre yapsak hep kriteri gevşetmek zorunda kalırdık.
        colour_match = (np.array([any(t.upper() in c for t in prefs.colours) for c in colours])
                        if prefs.colours else None)
        prices = self.items["price"].to_numpy()
        budget_mask = None
        if prefs.max_price:
            budget_mask = prices <= prefs.max_price
            if colour_match is not None:
                # İstenen renkteki ürün bütçeyi az aşıyorsa elemeyiz; "bütçenin biraz üstünde" diye gösteririz.
                budget_mask = budget_mask | (colour_match & (prices <= prefs.max_price * (1 + config.BUDGET_TOLERANCE)))
        masks = {
            "kategori": np.isin(cats, prefs.categories) if prefs.categories else None,
            "beden": np.array([(prefs.size in s) or not s for s in self.items["sizes"]]) if prefs.size else None,
            "bütçe": budget_mask,
        }
        if colour_match is not None:
            base = base + 2.0 * colour_match          # renge uyanlar listenin en üstüne çıkar
        active = [name for name, m in masks.items() if m is not None]
        relax_priority = ["beden", "bütçe", "kategori"]            # eşitlikte önce en önemsiz olanı gevşet
        notes: list[str] = []
        need = min(k, 3)                                            # en az bu kadar aday ürün kalmalı

        def combined(names):
            m = np.ones(len(self.items), dtype=bool)
            for name in names:
                m &= masks[name]
            return m

        mask = combined(active)
        while mask.sum() < need and active:
            # Darboğaz hangi kriter? Her kriteri tek tek çıkarıp en çok ürün kazandıranı gevşetiriz.
            # Kategoriyi, başka kriter kaldıysa korur (müşterinin asıl istediği o).
            options = [n for n in active if n != "kategori"] or active
            options.sort(key=relax_priority.index)
            drop = max(options, key=lambda n: combined([a for a in active if a != n]).sum())
            active.remove(drop)
            notes.append(f"'{drop}' kriterine uyan yeterli ürün bulunamadığı için bu kriter gevşetildi.")
            mask = combined(active)

        score = np.where(mask, base, -np.inf)
        top = [p for p in np.argsort(-score)[:k] if np.isfinite(score[p])]
        recs = []
        for p in top:
            reasons = []
            if seg_id is not None:
                reasons.append(f"Benzer tercihleri olan müşteri grubunda ('{self.seg.names[seg_id]}') bu ürünü alma oranı: %{self.seg_pop[seg_id][p] * 100:.1f}")
            else:
                reasons.append(f"Mağazamızın en çok tercih edilen ürünlerinden biri (tüm müşterilerde alma oranı: %{self.global_pop[p] * 100:.1f})")
            if colour_match is not None and colour_match[p]:
                reasons.append("İstediğiniz renk ailesine uyuyor.")
            if "bütçe" in active:
                price = self.items.iloc[p]["price"]
                if price <= prefs.max_price:
                    reasons.append(f"Tipik fiyatı yaklaşık {price:.0f}, bütçenizin içinde.")
                else:
                    reasons.append(f"Tipik fiyatı yaklaşık {price:.0f}; bütçenizin biraz üzerinde ama istediğiniz renkte.")
            if "beden" in active and prefs.size in self.items.iloc[p]["sizes"]:
                reasons.append(f"{prefs.size} bedeni bu üründe satılıyor.")
            recs.append(self._make_rec(p, score[p], reasons))
        if colour_match is not None and recs:
            n_match = sum(bool(colour_match[self._item_pos[r.item_id]]) for r in recs)
            if n_match == 0:
                notes.append("İstediğiniz renkte (diğer kriterlerle birlikte) ürün bulunamadı; en yakın alternatifler gösteriliyor.")
            elif n_match < len(recs):
                notes.append(f"İstediğiniz renkte yalnızca {n_match} ürün var; diğerleri alternatif olarak eklendi.")
        return RecommendationResult(recs, notes, seg_id, self.seg.names.get(seg_id) if seg_id is not None else None)

    # ------------------------------------------------------------------ paket önerisi (Apriori)
    def bundle_for(self, category: str, k: int = 3) -> list[Recommendation]:
        """'Bu kategoriyi alanlar şunları da aldı' – kategori düzeyindeki kurallardan.

        Kategorinin tek başına olduğu kuralların yanında, başka kategorilerle birlikte
        antecedent'te geçtiği kurallar da taranır (örn. 'gömlek + kazak -> ceket').
        Aynı sonuç kategorisi için en yüksek lift'li kural kullanılır.
        """
        if self.rules.empty:
            return []
        has = self.rules["antecedent"].apply(lambda a: category in a)
        r = self.rules[has & (self.rules["consequent"] != category)]
        r = r.sort_values("lift", ascending=False).drop_duplicates("consequent").head(k)
        recs = []
        for _, rule in r.iterrows():
            cand = self.items[self.items["category"] == rule["consequent"]]
            if cand.empty:
                continue
            pos = self._item_pos[cand.index[0]]            # katalog n_buyers'a göre sıralı: ilki en popüler
            names = " ve ".join(config.CATEGORY_TR.get(c, c) for c in rule["antecedent"])
            reason = (f"{names.capitalize()} alan müşterilerde "
                      f"{config.CATEGORY_TR.get(rule['consequent'], rule['consequent'])} alma oranı: %{rule['confidence'] * 100:.0f} "
                      f"(lift {rule['lift']:.2f}; 1'den büyük = tesadüften güçlü ilişki)")
            recs.append(self._make_rec(pos, rule["lift"], [reason]))
        return recs