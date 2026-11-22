"""DBSCAN-based song recommendation."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RECOMMENDATION_COUNT
from src.data_handling.features import find_song_index, prepare_feature_matrix


class DBSCANRecommender:
    """Recommend songs based on density-connected clusters."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    ):
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")

        self.eps = eps
        self.min_samples = min_samples
        self.recommendation_count = recommendation_count

        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler: StandardScaler | None = None
        self.songs: pd.DataFrame | None = None
        self.features: np.ndarray | None = None
        self.feature_columns: list[str] = []
        self.labels: np.ndarray | None = None

    def fit(self, songs: pd.DataFrame) -> "DBSCANRecommender":
        """Prepare song features and fit the DBSCAN clustering model."""
        if songs.empty:
            raise ValueError("Cannot fit DBSCAN with an empty dataset.")

        self.songs = songs.reset_index(drop=True).copy()
        self.features, self.scaler, self.feature_columns = (
            prepare_feature_matrix(self.songs)
        )

        self.labels = self.model.fit_predict(self.features)

        return self

    def find_song(self, track_id: str) -> dict:
        """Find a song and its cluster assignment."""
        if self.songs is None:
            raise RuntimeError("Call fit() before finding songs.")

        index = find_song_index(self.songs, track_id)

        if self.labels is None:
            raise NotImplementedError("DBSCAN cluster labels are not ready.")

        result = self.songs.iloc[index].to_dict()
        result["cluster"] = int(self.labels[index])
        return result

    def get_cluster(self, track_id: str) -> int:
        """Return the cluster label for a specific song."""
        song = self.find_song(track_id)
        return song["cluster"]

    def recommend(
        self,
        track_id: str,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Return songs from the same cluster as the requested song."""
        if self.songs is None:
            raise RuntimeError("Call fit() before requesting recommendations.")

        count = n if n is not None else self.recommendation_count
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        index = find_song_index(self.songs, track_id)

        if self.labels is None:
            raise NotImplementedError("DBSCAN cluster labels are not ready.")

        target_cluster = self.labels[index]

        if target_cluster == -1:
            return pd.DataFrame(columns=self.songs.columns)

        same_cluster_indices = np.where(self.labels == target_cluster)[0]
        same_cluster_indices = [i for i in same_cluster_indices if i != index]

        rec_indices = same_cluster_indices[:count]
        return self.songs.iloc[rec_indices].copy()
