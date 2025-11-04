from abc import ABC, abstractmethod
from typing import Sequence

import pandas as pd


class DataSource(ABC):
    """Abstract base class for all data sources that produce fight-level features."""

    def __init__(
        self,
        source_id: str,
        join_keys: Sequence[str],
        feature_prefix: str | None = None,
    ) -> None:
        self._source_id = source_id
        self._join_keys = tuple(join_keys)
        if not self._join_keys:
            raise ValueError("DataSource requires at least one join key.")
        self._feature_prefix = feature_prefix or source_id

    @property
    def id(self) -> str:
        """Human-readable identifier used in logs and dependency declarations."""
        return self._source_id

    @property
    def join_keys(self) -> tuple[str, ...]:
        """Column names that uniquely identify rows for joins."""
        return self._join_keys

    @property
    def feature_prefix(self) -> str:
        """Prefix applied to overlapping feature names to avoid collisions."""
        return self._feature_prefix

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Return a DataFrame containing the features this data source owns."""

    def augment(self, base_df: pd.DataFrame, how: str = "left") -> pd.DataFrame:
        """Merge the data source features into the provided DataFrame."""
        self._assert_join_keys(base_df)
        features = self.load()
        self._assert_join_keys(features, context="features DataFrame")

        merged_features = self._dedupe_feature_columns(base_df, features)
        return base_df.merge(
            merged_features,
            how=how,
            on=list(self.join_keys),
            validate="one_to_one",
        )

    def _assert_join_keys(self, df: pd.DataFrame, context: str = "base DataFrame") -> None:
        missing = [key for key in self.join_keys if key not in df.columns]
        if missing:
            raise KeyError(f"{context} missing join keys required by {self.id}: {missing}")

    def _dedupe_feature_columns(
        self,
        base_df: pd.DataFrame,
        feature_df: pd.DataFrame,
    ) -> pd.DataFrame:
        overlap = set(base_df.columns) & set(feature_df.columns)
        overlap -= set(self.join_keys)
        if not overlap:
            return feature_df

        rename_mapping: dict[str, str] = {
            column: f"{self.feature_prefix}_{column}" for column in overlap
        }
        return feature_df.rename(columns=rename_mapping)
