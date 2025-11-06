from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Sequence

import pandas as pd

from .base import DataSource


class BaseFightsDataSource(DataSource):
    """Data source that provides the canonical fight records including outcomes."""

    class OutputDFColumns(StrEnum):
        FIGHT_ID = "fight_id"
        FIGHT_URL = "fight_url"
        EVENT_ID = "event_id"
        EVENT_DATE = "event_date"
        RED_ID = "red_id"
        RED_NAME = "red_name"
        BLUE_ID = "blue_id"
        BLUE_NAME = "blue_name"
        REFEREE = "referee"
        OUTCOME = "outcome"

    class Outcome(IntEnum):
        RED_WIN = 0
        BLUE_WIN = 1
        DRAW_NO_CONTEST = 2

    def __init__(
        self,
        csv_path: Path,
        *,
        join_keys: Sequence[str] = (OutputDFColumns.FIGHT_ID,),
        outcome_column_name: str = "outcome",
    ) -> None:
        super().__init__(
            source_id="base_fights", join_keys=join_keys, feature_prefix="base"
        )
        self._csv_path = csv_path
        self._outcome_column_name = outcome_column_name

    def load(self) -> pd.DataFrame:
        """Load the fight data from disk."""
        df = pd.read_csv(self._csv_path)
        df[self._outcome_column_name] = df.apply(self._map_outcome, axis=1)
        valid_columns = {col.value for col in self.OutputDFColumns}
        df = df[[col for col in df.columns if col in valid_columns]]
        return df

    def _map_outcome(self, row: pd.Series) -> int | None:
        red = (row.get("red_result") or "").strip().upper()
        blue = (row.get("blue_result") or "").strip().upper()
        if red.startswith("W"):
            return BaseFightsDataSource.Outcome.RED_WIN
        if blue.startswith("W"):
            return BaseFightsDataSource.Outcome.BLUE_WIN
        return BaseFightsDataSource.Outcome.DRAW_NO_CONTEST
