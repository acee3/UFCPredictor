from dataclasses import dataclass
from typing import Tuple, Union

from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class TrainingResult:
    trained_pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
