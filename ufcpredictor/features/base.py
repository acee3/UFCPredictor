from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd


class FeatureBuilder(Protocol):
    """Interface for engineered feature builders."""

    @property
    def id(self) -> str:  # pragma: no cover - Protocol definition
        ...

    @property
    def required_sources(self) -> Sequence[str]:  # pragma: no cover - Protocol definition
        ...

    @property
    def required_features(self) -> Sequence[str]:  # pragma: no cover - Protocol definition
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

