from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from lab3.constants import CHURN, D_NUMERIC, DATA_FILE, DIAMONDS_DATA_FILE, NUMERIC_COLUMNS
except ModuleNotFoundError:
    from constants import CHURN, D_NUMERIC, DATA_FILE, DIAMONDS_DATA_FILE, NUMERIC_COLUMNS

TEST_SIZE = 0.2
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parents[1]


def _resolve_data_path(path: Path) -> Path:
    candidates = [
        BASE_DIR / path,
        BASE_DIR.parent / "lab2" / path,
        BASE_DIR.parent / "lab1" / path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_telco_data() -> pd.DataFrame:
    data = pd.read_csv(_resolve_data_path(DATA_FILE))

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def split_telco_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return tuple(
        train_test_split(
            data,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=data[CHURN],
        )
    )


def load_diamonds_data() -> pd.DataFrame:
    diamonds_df = pd.read_csv(_resolve_data_path(DIAMONDS_DATA_FILE))

    for column in D_NUMERIC:
        diamonds_df[column] = pd.to_numeric(diamonds_df[column], errors="coerce")

    return diamonds_df


def split_diamonds_data(diamonds_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return tuple(
        train_test_split(
            diamonds_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )
