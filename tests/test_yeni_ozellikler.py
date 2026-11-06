"""Mevsim etkisi, sepet tamamlama ve ayrıştırıcı doğruluk testi için testler."""
import numpy as np

import config
from src.bot import StyleBot, format_bundle
from src.llm import LLMClient
from src.parser_testi import CASES, run
from src.recommender import Preferences


def test_season_boost_is_neutral_without_month_and_bounded_with_month(artifacts):
    rec = artifacts.recommender
    assert np.all(rec.season_boost(None) == 1.0)
    b = rec.season_boost(1)
    w = config.SEASON_WEIGHT
    assert len(b) == len(rec.items)
    assert b.min() >= 1 - w * 0.5 - 1e-9 and b.max() <= 1 + w * 1.0 + 1e-9    # çarpan 0.5-2 arasına kırpılmıştı


def test_season_only_reorders_never_breaks_filters(artifacts):
    rec = artifacts.recommender
    prefs = Preferences(categories=["SHIRTS"], max_price=3000)
    for month in (1, 7):
        res = rec.recommend_new(prefs, month=month)
        assert res.recs and all(r.category == "SHIRTS" and r.price <= 3000 for r in res.recs)
    uid = rec.user_ids[0]
    owned = {rec.item_ids[p] for p in rec.user_item[rec._user_pos[uid]].indices}
    assert not owned & {r.item_id for r in rec.recommend_existing(uid, month=3).recs}


def test_season_weight_zero_gives_same_result_as_no_season(artifacts, monkeypatch):
    rec = artifacts.recommender
    monkeypatch.setattr(config, "SEASON_WEIGHT", 0.0)
    a = [r.item_id for r in rec.recommend_new(Preferences(categories=["SHIRTS"]), month=2).recs]
    b = [r.item_id for r in rec.recommend_new(Preferences(categories=["SHIRTS"])).recs]
    assert a == b


def test_bot_adds_bundle_when_rules_exist(artifacts, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bot = StyleBot(artifacts, LLMClient())
    cats = [c for c in sorted(set(artifacts.recommender.items["category"])) if artifacts.recommender.bundle_for(c, k=2)]
    assert cats, "sentetik veride en az bir kategori için birliktelik kuralı olmalı"
    c_tr = config.CATEGORY_TR[cats[0]]
    resp = bot.respond(f"{c_tr} arıyorum")
    assert resp.bundle and "Yanına şunlar da gidebilir" in format_bundle(resp.bundle)
    assert "Yanına şunlar da gidebilir" in bot.reply(f"{c_tr} arıyorum")


def test_parser_accuracy_report_is_consistent():
    res = run()
    assert res["cumle_sayisi"] == len(CASES)
    assert all(0 <= v <= 1 for v in res["dogruluk"].values())
    assert res["dogruluk"]["kategori"] >= 0.8          # çok düşerse ayrıştırıcı bozulmuş demektir
