import config
from src import cache


def test_cache_roundtrip_and_invalidation(artifacts, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    f = tmp_path / "d.csv"
    f.write_text("a")
    sig = cache.signature([f])
    cache.save(sig, artifacts)
    loaded = cache.load(sig)
    assert loaded is not None and loaded.df is None            # büyük tablo kaydedilmez
    uid = artifacts.recommender.user_ids[0]
    a = [r.item_id for r in artifacts.recommender.recommend_existing(uid).recs]
    b = [r.item_id for r in loaded.recommender.recommend_existing(uid).recs]
    assert a == b                                              # yüklenen model aynı öneriyi verir
    f.write_text("değişti, boyut farklı")
    assert cache.load(cache.signature([f])) is None            # veri değişince önbellek geçersiz
