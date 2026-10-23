import numpy as np
import pandas as pd

from src import tekrar_gelme


def _toy(n_users=600, seed=0):
    """Kural: son zamanlarda ve sık alışveriş yapanlar geri gelir. Model bunu öğrenebilmeli."""
    rng = np.random.default_rng(seed)
    rows = []
    end = pd.Timestamp("2022-12-31")
    for u in range(n_users):
        loyal = rng.random() < 0.5
        n_bills = rng.integers(4, 9) if loyal else rng.integers(1, 3)
        for b in range(n_bills):
            d = pd.Timestamp("2022-01-01") + pd.Timedelta(days=int(rng.integers(0, 170 if not loyal else 360)))
            rows.append({"user_id": f"U{u}", "bill_id": f"B{u}_{b}", "transaction_date": d, "amount": 1000.0,
                         "bill_discount": 0.0, "category": "SHIRTS"})
        if loyal:                                          # sadık müşteri kesimden sonra da gelir
            rows.append({"user_id": f"U{u}", "bill_id": f"B{u}_x", "transaction_date": end - pd.Timedelta(days=int(rng.integers(1, 100))),
                         "amount": 1000.0, "bill_discount": 0.0, "category": "SHIRTS"})
    return pd.DataFrame(rows)


def test_labels_use_only_future_and_features_only_past():
    df = _toy()
    X, y, cutoff = tekrar_gelme.build_dataset(df, horizon_days=180)
    assert cutoff == df["transaction_date"].max() - pd.Timedelta(days=180)
    past = df[df["transaction_date"] <= cutoff]
    assert set(X.index) == set(past["user_id"])                   # yalnızca geçmişi olan müşteriler
    uid = X.index[0]
    assert X.loc[uid, "frequency"] == past[past["user_id"] == uid]["bill_id"].nunique()   # özellik sadece geçmişten
    future = set(df[df["transaction_date"] > cutoff]["user_id"])
    assert all(y[u] == int(u in future) for u in X.index[:50])


def test_models_beat_baseline_on_learnable_pattern():
    res = tekrar_gelme.train_and_evaluate(_toy(1200), horizon_days=180)
    assert res["lojistik_regresyon"]["roc_auc"] > 0.8
    assert res["rastgele_orman"]["roc_auc"] > 0.8
    assert res["rastgele_orman"]["roc_auc"] > res["referans"]["roc_auc"] - 0.01
    assert not res["onem"].empty and res["ozet"]["n_test"] > 0


def test_single_class_returns_note():
    df = _toy(200)
    df = df[df["transaction_date"] <= df["transaction_date"].max() - pd.Timedelta(days=180)]   # gelecek yok
    res = tekrar_gelme.train_and_evaluate(df)
    assert "not" in res
