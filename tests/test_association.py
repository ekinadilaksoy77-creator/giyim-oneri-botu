import pytest

from src.association import association_rules, frequent_itemsets


def test_apriori_hand_computed():
    # 5 sepet: A ve B 3 kez birlikte; A toplam 4 kez; B toplam 4 kez
    baskets = [{"A", "B"}, {"A", "B"}, {"A", "B", "C"}, {"A"}, {"B", "C"}]
    freq, n = frequent_itemsets(baskets, min_support=0.4, max_len=3)
    assert n == 5
    assert freq[frozenset("A")] == 4
    assert freq[frozenset("AB")] == 3
    rules = association_rules(freq, n, min_confidence=0.0, min_lift=0.0)
    r = rules[(rules["antecedent"] == ("A",)) & (rules["consequent"] == "B")].iloc[0]
    assert r["support"] == pytest.approx(3 / 5)
    assert r["confidence"] == pytest.approx(3 / 4)        # A alanların 3/4'ü B de aldı
    assert r["lift"] == pytest.approx((3 / 4) / (4 / 5))  # confidence / support(B)


def test_min_support_filters_rare_items():
    baskets = [{"A", "B"}] * 9 + [{"Z"}]
    freq, _ = frequent_itemsets(baskets, min_support=0.2)
    assert frozenset("Z") not in freq


def test_triples_need_all_subsets_frequent():
    baskets = [{"A", "B", "C"}] * 4 + [{"A", "B"}] * 2
    freq, _ = frequent_itemsets(baskets, min_support=0.5, max_len=3)
    assert frozenset("ABC") in freq and freq[frozenset("ABC")] == 4
