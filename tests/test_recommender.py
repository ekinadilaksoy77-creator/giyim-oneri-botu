import pytest

import config
from src.llm import LLMClient, rule_based_preferences
from src.recommender import Preferences, UnknownCustomerError


def test_each_customer_gets_exactly_one_segment(artifacts):
    assert artifacts.seg.labels.notna().all()
    assert set(artifacts.seg.labels) == set(artifacts.seg.names)


def test_existing_customer_never_recommended_owned_items(artifacts):
    rec = artifacts.recommender
    uid = rec.user_ids[0]
    owned = {rec.item_ids[p] for p in rec.user_item[rec._user_pos[uid]].indices}
    res = rec.recommend_existing(uid, k=5)
    assert 0 < len(res.recs) <= 5
    assert not owned & {r.item_id for r in res.recs}


def test_unknown_customer_raises(artifacts):
    with pytest.raises(UnknownCustomerError):
        artifacts.recommender.recommend_existing("YOK-BOYLE-BIR-MUSTERI")


def test_new_customer_filters_by_category_and_budget(artifacts):
    res = artifacts.recommender.recommend_new(Preferences(categories=["SHIRTS"], max_price=3000))
    assert res.recs and all(r.category == "SHIRTS" for r in res.recs)
    assert all(r.price <= 3000 for r in res.recs)
    assert res.segment_id is not None


def test_new_customer_always_gets_a_recommendation_even_with_impossible_filters(artifacts):
    prefs = Preferences(categories=["SHIRTS"], colours=["PEMBE"], size="XXL", max_price=1)
    res = artifacts.recommender.recommend_new(prefs)
    assert len(res.recs) >= 1          # kabul kriteri: yeni müşteriye her zaman öneri dönmeli
    assert res.notes                   # ve kriterlerin gevşetildiği söylenmeli


def test_relaxes_only_the_bottleneck_filter(artifacts):
    # Sorun yalnızca bütçede (1 birim): kategori ve renk korunmalı, not 'bütçe'yi söylemeli.
    res = artifacts.recommender.recommend_new(Preferences(categories=["SHIRTS"], colours=["BLACK"], max_price=1))
    assert res.recs and all(r.category == "SHIRTS" for r in res.recs)
    assert "BLACK" in res.recs[0].colour                   # istenen renk en üstte
    assert any("bütçe" in n for n in res.notes)            # gevşetilen kriter doğru söylenmiş
    assert not any("'kategori'" in n or "'beden'" in n for n in res.notes)


def test_empty_preferences_fall_back_to_popular(artifacts):
    res = artifacts.recommender.recommend_new(Preferences())
    assert len(res.recs) == config.DEFAULT_TOP_K
    assert [r.item_id for r in res.recs] == artifacts.recommender.popular_items(config.DEFAULT_TOP_K)


def test_recommendations_only_from_catalog(artifacts):
    rec = artifacts.recommender
    res = rec.recommend_new(Preferences(categories=["JACKETS"]))
    assert {r.item_id for r in res.recs} <= set(rec.item_ids)


def test_rule_based_parser_turkish():
    p = rule_based_preferences("İş görüşmesi için lacivert bir gömlek arıyorum, M beden, bütçem 1.500")
    assert p.categories == ["SHIRTS"]
    assert "NAVY" in p.colours
    assert p.size == "M"
    assert p.max_price == 1500


def test_llm_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm = LLMClient()
    assert not llm.available
    p = llm.extract_preferences("siyah kemer lazım", ["BLACK"])
    assert p.categories == ["BELTS"] and "BLACK" in p.colours


def test_unknown_colour_items_never_recommended(artifacts):
    rec = artifacts.recommender
    assert not any(config.NOT_CAPTURED in i for i in rec.item_ids)
    res = rec.recommend_new(Preferences(categories=["SHIRTS"]))
    assert all(r.colour != config.NOT_CAPTURED for r in res.recs)


def test_requested_colour_kept_when_slightly_over_budget(artifacts):
    rec = artifacts.recommender
    shirts = rec.items[rec.items["category"] == "SHIRTS"]
    row = shirts.loc[shirts["price"].idxmax()]              # en pahalı gömlek: diğerleri bütçeye uyar, bu %10 aşar
    slightly_below = row["price"] / 1.1                       # ürün bütçeyi %10 aşıyor
    res = rec.recommend_new(Preferences(categories=["SHIRTS"], colours=[row["colour"]], max_price=slightly_below))
    assert res.recs[0].item_id == row.name                    # istenen renk elenmedi
    assert any("biraz üzerinde" in r for r in res.recs[0].reasons)


def test_bundle_has_no_duplicate_categories_and_excludes_itself(artifacts):
    rec = artifacts.recommender
    for cat in sorted(set(rec.items["category"])):
        out = rec.bundle_for(cat, k=5)
        cats = [r.category for r in out]
        assert cat not in cats and len(cats) == len(set(cats))