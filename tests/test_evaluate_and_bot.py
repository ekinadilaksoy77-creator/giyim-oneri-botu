from src.bot import StyleBot
from src.evaluate import evaluate, temporal_split
from src.llm import LLMClient


def test_temporal_split_has_no_leakage(clean_df):
    train, test, cutoff = temporal_split(clean_df)
    assert train["transaction_date"].max() <= cutoff < test["transaction_date"].min()


def test_evaluate_runs_and_hybrid_beats_global_popularity(clean_df):
    res = evaluate(clean_df, k=5, max_users=500)
    m = res["mevcut_musteri"]
    assert m["n_musteri"] > 30
    for method in ("populer", "segment_populer", "benzer_urun", "hibrit"):
        assert 0 <= m[method]["hit_rate@k"] <= 1
    # Sentetik veride müşteri tipleri belirgin olduğu için kişiselleştirme popülerliği geçmeli.
    assert m["hibrit"]["hit_rate@k"] >= m["populer"]["hit_rate@k"]


def test_bot_end_to_end_without_llm(artifacts, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    bot = StyleBot(artifacts, LLMClient())
    out = bot.reply("lacivert gömlek arıyorum, bütçem 3000")
    assert "Gömlek" in out and "•" in out
    uid = artifacts.recommender.user_ids[0]
    assert "tipik fiyat" in bot.reply(f"müşteri {uid}")
    assert "kayıtlarda yok" in bot.reply("müşteri HAYALET")
    assert "anlayamadım" in bot.reply("merhaba")


def test_exit_shortcuts():
    from src.bot import is_exit
    for w in ["çık", "ÇIK", "Çıkış", "q", "EXIT", " quit "]:
        assert is_exit(w), w
    assert not is_exit("çıkış yapmayan gömlek arıyorum")
