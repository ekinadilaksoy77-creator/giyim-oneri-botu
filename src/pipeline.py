"""Tüm adımları tek yerde birleştirir: veri -> özellik -> segment -> kurallar -> öneri motoru."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.association import mine_rules
from src.features import build_customer_features
from src.recommender import HybridRecommender
from src.segmentation import SegmentationResult, fit_segments


@dataclass
class Artifacts:
    df: pd.DataFrame
    features: pd.DataFrame
    seg: SegmentationResult
    rules: pd.DataFrame
    recommender: HybridRecommender


def build(df: pd.DataFrame, k: int | None = None, min_support: float = 0.002) -> Artifacts:
    features = build_customer_features(df)
    seg = fit_segments(features, k=k)
    rules = mine_rules(df, level="category", min_support=min_support)
    rec = HybridRecommender(df, seg, rules)
    return Artifacts(df, features, seg, rules, rec)