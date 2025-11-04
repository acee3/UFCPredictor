from dataclasses import dataclass
from typing import Tuple, Union

from sklearn.pipeline import Pipeline
import pandas as pd


@dataclass(frozen=True)
class BaseFightInput:
    """Minimal information required to identify a fight and its outcome."""

    event_id: str
    fighter_ids: Tuple[Union[int, str], Union[int, str]]
    outcome: Union[int, float, str]

    def to_record(self) -> dict[str, Union[int, float, str]]:
        """Flatten the dataclass into a DataFrame-friendly record."""
        fighter_a, fighter_b = self.fighter_ids
        return {
            "event_id": self.event_id,
            "fighter_id_a": fighter_a,
            "fighter_id_b": fighter_b,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class TrainingResult:
    trained_pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_output: pd.DataFrame
