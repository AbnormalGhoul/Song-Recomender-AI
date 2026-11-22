"""K-nearest-neighbor song recommendation."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RECOMMENDATION_COUNT
from src.data_handling.features import find_song_index, prepare_feature_matrix


class KNNRecommender:
    """Recommend songs based on feature-space distance."""

    def __init__(
        self,
        n_neighbors: int = 10,
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    ):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")

        self.n_neighbors = n_neighbors
        self.recommendation_count = recommendation_count
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
        )
        self.scaler: StandardScaler | None = None
        self.songs: pd.DataFrame | None = None
        self.features: np.ndarray | None = None
        self.feature_columns: list[str] = []

    def fit(self, songs: pd.DataFrame) -> "KNNRecommender":
        """Prepare features and fit the nearest-neighbor model."""
        if songs.empty:
            raise ValueError("Cannot fit KNN with an empty dataset.")

        self.songs = songs.reset_index(drop=True).copy()
        self.features, self.scaler, self.feature_columns = (
            prepare_feature_matrix(self.songs)
        )

        self.model.fit(self.features)

        return self

    def find_song(self, track_id: str) -> dict:
        """Return a song record identified by its track ID."""
        if self.songs is None:
            raise RuntimeError("Call fit() before finding songs.")

        index = find_song_index(self.songs, track_id)
        return self.songs.iloc[index].to_dict()

    def recommend(
        self,
        track_id: str,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Return the nearest songs, excluding the input song itself."""
        if self.songs is None or self.features is None:
            raise RuntimeError("Call fit() before requesting recommendations.")

        count = n if n is not None else self.recommendation_count
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        index = find_song_index(self.songs, track_id)

        distances, indices = self.model.kneighbors(
            [self.features[index]],
            n_neighbors=min(count + 1, len(self.features)),
        )

        rec_indices = [i for i in indices[0] if i != index][:count]

        return self.songs.iloc[rec_indices].copy()
