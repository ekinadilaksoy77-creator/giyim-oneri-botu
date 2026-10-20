import pytest

from src import data_loader, synthetic_data
from src.pipeline import build


@pytest.fixture(scope="session")
def raw_df():
    return synthetic_data.generate(n_users=1500, seed=7)


@pytest.fixture(scope="session")
def clean_df(raw_df):
    df, _ = data_loader.clean(raw_df)
    return df


@pytest.fixture(scope="session")
def artifacts(clean_df):
    return build(clean_df, k=4)
