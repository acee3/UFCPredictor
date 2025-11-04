from typing import Sequence

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .data_sources import DataSource
from .features import FeatureBuilder
from .splitters import TrainTestSplitStrategy
from .types import BaseFightInput, TrainingResult


def run_pipeline(
    base_inputs: Sequence[BaseFightInput],
    data_sources: Sequence[DataSource],
    engineered_features: Sequence[FeatureBuilder],
    splitter: TrainTestSplitStrategy,
    model: BaseEstimator,
) -> TrainingResult:
    """Assemble, fit, and return an sklearn pipeline based on configurable components."""
    if not base_inputs:
        raise ValueError("base_inputs must contain at least one record.")

    base_df = _build_base_dataframe(base_inputs)
    feature_df = _merge_data_sources(base_df, data_sources)
    feature_df = _apply_feature_builders(feature_df, data_sources, engineered_features)

    if "outcome" not in feature_df.columns:
        raise KeyError("The consolidated feature DataFrame must contain an 'outcome' column.")

    y = feature_df["outcome"]
    X = feature_df.drop(columns=["outcome"])

    X_train, X_test, y_train, y_test = _split_data(splitter, X, y)

    pipeline = Pipeline(steps=[("model", model)])
    pipeline.fit(X_train, y_train)
    
    test_predictions = pipeline.predict(X_test)
    test_output = pd.DataFrame({"prediction": test_predictions}, index=y_test.index.copy())
    test_output["actual"] = y_test

    return TrainingResult(
        trained_pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        test_output=test_output,
    )


def _build_base_dataframe(base_inputs: Sequence[BaseFightInput]) -> pd.DataFrame:
    records = [base_input.to_record() for base_input in base_inputs]
    return pd.DataFrame.from_records(records)


def _merge_data_sources(
    base_df: pd.DataFrame,
    data_sources: Sequence[DataSource],
) -> pd.DataFrame:
    merged = base_df.copy()
    for data_source in data_sources:
        merged = data_source.augment(merged)
    return merged


def _apply_feature_builders(
    df: pd.DataFrame,
    data_sources: Sequence[DataSource],
    engineered_features: Sequence[FeatureBuilder],
) -> pd.DataFrame:
    available_sources = {source.id for source in data_sources}
    produced_features = set()
    transformed = df.copy()

    for builder in engineered_features:
        builder_id = _validate_feature_dependencies(
            builder=builder,
            available_sources=available_sources,
            produced_features=produced_features,
        )
        transformed = builder.transform(transformed)
        produced_features.add(builder_id)

    return transformed


def _validate_feature_dependencies(
    *,
    builder: FeatureBuilder,
    available_sources: set[str],
    produced_features: set[str],
) -> str:
    builder_id = getattr(builder, "id", builder.__class__.__name__)
    required_sources = set(getattr(builder, "required_sources", []))
    required_features = set(getattr(builder, "required_features", []))

    _raise_if_missing(
        missing=required_sources - available_sources,
        builder_id=builder_id,
        error_template="requires missing data sources: {missing}",
    )
    _raise_if_missing(
        missing=required_features - produced_features,
        builder_id=builder_id,
        error_template="depends on features that have not been created yet: {missing}",
    )

    return builder_id


def _raise_if_missing(
    *,
    missing: set[str],
    builder_id: str,
    error_template: str,
) -> None:
    if not missing:
        return
    formatted_missing = ", ".join(sorted(missing))
    raise ValueError(f"Feature builder '{builder_id}' {error_template.format(missing=formatted_missing)}")


def _split_data(
    splitter: TrainTestSplitStrategy,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if hasattr(splitter, "split"):
        return splitter.split(X, y)  # type: ignore[call-arg]
    return splitter(X, y)
