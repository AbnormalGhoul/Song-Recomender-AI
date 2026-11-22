"""Shared feature preparation for recommendation algorithms."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import SONG_FEATURES


def get_feature_columns(
    songs: pd.DataFrame,
    requested_features: Sequence[str] | None = None,
) -> list[str]:
    """Return requested numeric features available in the dataset."""
    candidates = list(requested_features or SONG_FEATURES)

    columns = [
        name for name in candidates
        if name in songs.columns
        and pd.api.types.is_numeric_dtype(songs[name])
    ]

    if not columns:
        raise ValueError("No usable numeric song features were found.")

    return columns


def prepare_feature_matrix(
    songs: pd.DataFrame,
    requested_features: Sequence[str] | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Clean, impute, and standardize numeric song features."""
    if songs.empty:
        raise ValueError("Cannot prepare features from an empty dataset.")

    columns = get_feature_columns(songs, requested_features)
    numeric = songs[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    columns = [
        name for name in columns
        if numeric[name].notna().any()
    ]
    numeric = numeric[columns]

    if numeric.empty:
        raise ValueError("No usable numeric feature values were found.")

    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    values = numeric.to_numpy(dtype=np.float64)

    fitted_scaler = scaler or StandardScaler()
    matrix = (
        fitted_scaler.fit_transform(values)
        if scaler is None
        else fitted_scaler.transform(values)
    )

    return matrix, fitted_scaler, columns


def find_song_index(songs: pd.DataFrame, song_id: str) -> int:
    """Find a song by song_id, falling back to track_id."""
    for field in ("song_id", "track_id"):
        if field not in songs.columns:
            continue

        matches = np.flatnonzero(
            songs[field].astype(str).to_numpy() == str(song_id)
        )

        if len(matches):
            return int(matches[0])

    raise ValueError(f"Song not found: {song_id}")
