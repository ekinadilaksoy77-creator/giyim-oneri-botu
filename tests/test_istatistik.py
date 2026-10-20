import numpy as np
import pandas as pd

from src import istatistik as ist


def _monthly_df(n_months=48, slope=1000.0, season_amp=5000.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_months):
        date = pd.Timestamp("2019-01-15") + pd.DateOffset(months=i)
        rev = 100_000 + slope * i + season_amp * (1 if date.month in (11, 12) else 0) + rng.normal(0, noise)
        rows.append({"transaction_date": date, "amount": rev})
    return pd.DataFrame(rows)


def test_regression_recovers_trend_and_seasonality():
    r = ist.revenue_regression(_monthly_df(slope=1000.0, season_amp=5000.0))
    o = r["ozet"]
    assert abs(o["trend_katsayisi_aylik"] - 1000.0) < 1.0        # gürültüsüz veride eğim birebir bulunur
    assert o["r2"] > 0.999 and o["trend_p"] < 0.001
    assert r["katsayilar"].loc["ay_12", "katsayi"] > 4000          # Aralık etkisi yakalanır
    assert len(r["tahmin"]) == 6 and o["holdout_ay"] == 6
    assert o["holdout_mape"] < 0.01


def test_regression_short_series_has_no_seasonal_terms():
    r = ist.revenue_regression(_monthly_df(n_months=12))
    assert not r["ozet"]["mevsimsel_model"] and list(r["katsayilar"].index) == ["sabit", "trend"]
    assert r["ozet"]["holdout_ay"] == 0


def test_segment_tests_on_real_pipeline(artifacts, clean_df):
    res = ist.segment_spend_tests(artifacts.features, artifacts.seg.labels)
    assert res["grup_sayisi"] >= 2 and 0 <= res["eta_kare"] <= 1
    assert 0 <= res["anova_p"] <= 1 and 0 <= res["kruskal_p"] <= 1
    z = ist.segment_zone_test(clean_df, artifacts.seg.labels)
    assert 0 <= z["cramer_v"] <= 1 and 0 <= z["p"] <= 1


def test_effect_size_wording():
    assert ist.yorum_etki(0.005, "eta") == "ihmal edilebilir"
    assert ist.yorum_etki(0.2, "eta") == "büyük"
    assert ist.yorum_etki(0.2, "v") == "küçük"


def test_missing_month_does_not_shift_time_axis():
    df = _monthly_df(n_months=48, slope=1000.0, season_amp=0.0)
    df = df[df["transaction_date"].dt.to_period("M") != pd.Period("2020-04")]      # bir ay eksik
    r = ist.revenue_regression(df)
    assert r["ozet"]["eksik_aylar"] == ["2020-04"] and r["ozet"]["n_ay"] == 47
    assert abs(r["ozet"]["trend_katsayisi_aylik"] - 1000.0) < 1.0                 # eksik ay eğimi bozmaz