"""Data preprocessing for time series CSV files."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

INVALID_TOKENS = {"#DIV/0!", "#N/A", "#VALUE!", "#REF!", "#NAME?", "#NUM!", "#NULL!"}


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Convert series to numeric, treating invalid tokens as NaN."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.replace(list(INVALID_TOKENS), np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def _normalize_scada_timestamp(value: str) -> str:
    """Normalize SCADA timestamps like ``01/12/2019 00.10.00``."""
    import re

    text = str(value).strip()
    if re.match(r"^\d{2}/\d{2}/\d{4}$", text):
        return f"{text} 00:00:00"
    return re.sub(r"(\d{2})\.(\d{2})\.(\d{2})$", r"\1:\2:\3", text)


def _parse_datetime_column(series: pd.Series, data_cfg: dict[str, Any]) -> pd.Series:
    """Parse datetime column using optional profile-specific rules."""
    parse_mode = str(data_cfg.get("datetime_parse", "")).lower()
    if parse_mode == "scada_eu":
        normalized = series.map(_normalize_scada_timestamp)
        return pd.to_datetime(normalized, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_and_preprocess(cfg: dict[str, Any], project_root: str | None = None) -> pd.DataFrame:
    """
    Load CSV, clean invalid values, sort by datetime, and optionally create labels.

    Returns a DataFrame with feature columns, target column, and optional label column.
    """
    data_cfg = cfg["data"]
    path = data_cfg["path"]
    if project_root is not None:
        from pathlib import Path

        path = str(Path(project_root) / path)

    df = pd.read_csv(
        path,
        sep=data_cfg.get("csv_sep", ","),
        decimal=data_cfg.get("csv_decimal", "."),
        encoding=data_cfg.get("encoding", "utf-8"),
    )
    df.columns = [str(col).strip() for col in df.columns if str(col).strip()]

    datetime_col = data_cfg.get("datetime_col")
    if datetime_col and datetime_col in df.columns:
        df[datetime_col] = _parse_datetime_column(df[datetime_col], data_cfg)
        df = df.sort_values(datetime_col).reset_index(drop=True)

    feature_cols = list(data_cfg["feature_cols"])
    target_col = data_cfg.get("target_col")

    numeric_cols = feature_cols.copy()
    if target_col:
        numeric_cols.append(target_col)

    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric_series(df[col])

    fill_method = data_cfg.get("fill_method", "ffill")
    if fill_method == "ffill":
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
    elif fill_method == "drop":
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    else:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    task_type = cfg["task"]["type"]
    classify_cfg = data_cfg.get("classify", {})
    if task_type == "classify" or classify_cfg.get("enabled", False):
        if target_col is None:
            raise ValueError("target_col is required to generate classification labels.")
        thresholds = classify_cfg.get("thresholds", [])
        df["label"] = _bin_target(df[target_col], thresholds)

    return df


def _bin_target(target: pd.Series, thresholds: list[float]) -> pd.Series:
    """Bin continuous target into class labels using thresholds."""
    bins = [-np.inf] + list(thresholds) + [np.inf]
    labels = pd.cut(target, bins=bins, labels=False, include_lowest=True)
    return labels.astype(int)


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically into train/val/test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df
