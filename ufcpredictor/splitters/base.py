from __future__ import annotations

from typing import Protocol, Tuple

import pandas as pd


class TrainTestSplitStrategy(Protocol):
    """Strategy that splits feature matrices and labels into train/test sets."""

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...
