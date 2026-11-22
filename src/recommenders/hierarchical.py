"""Hierarchical-clustering song recommendation."""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RECOMMENDATION_COUNT
from src.data_handling.features import find_song_index, prepare_feature_matrix


class HierarchicalRecommender:
    """Recommend songs based on agglomerative clustering."""

    def __init__(
        self,
        n_clusters: int = 10,
        linkage: str = "ward",
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive.")
        if linkage not in {"ward", "complete", "average", "single"}:
            raise ValueError("Unsupported linkage method.")

        self.n_clusters = n_clusters
        self.linkage = linkage
        self.recommendation_count = recommendation_count

        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )
        self.scaler: StandardScaler | None = None
        self.songs: pd.DataFrame | None = None
        self.features: np.ndarray | None = None
        self.feature_columns: list[str] = []
        self.labels: np.ndarray | None = None

    def fit(self, songs: pd.DataFrame) -> "HierarchicalRecommender":
        """Prepare song features and fit the clustering model."""
        if songs.empty:
            raise ValueError("Cannot fit hierarchical clustering on no songs.")
        if len(songs) < self.n_clusters:
            raise ValueError(
                "The number of songs must be at least n_clusters."
            )

        self.songs = songs.reset_index(drop=True).copy()
        self.features, self.scaler, self.feature_columns = (
            prepare_feature_matrix(self.songs)
        )

        self.labels = self.model.fit_predict(self.features)

        return self

    def find_song(self, track_id: str) -> dict:
        """Find a song and return its cluster assignment."""
        if self.songs is None:
            raise RuntimeError("Call fit() before finding songs.")

        index = find_song_index(self.songs, track_id)

        if self.labels is None:
            raise NotImplementedError("Cluster labels are not ready.")

        result = self.songs.iloc[index].to_dict()
        result["cluster"] = int(self.labels[index])
        return result

    def get_cluster(self, track_id: str) -> int:
        """Return the cluster label for a song."""
        return self.find_song(track_id)["cluster"]

    def recommend(
        self,
        track_id: str,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Return other songs from the same hierarchical cluster."""
        if self.songs is None:
            raise RuntimeError("Call fit() before requesting recommendations.")

        count = n if n is not None else self.recommendation_count
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        index = find_song_index(self.songs, track_id)

        if self.labels is None:
            raise NotImplementedError("Cluster labels are not ready.")

        target_cluster = self.labels[index]
        same_cluster_indices = np.where(self.labels == target_cluster)[0]
        same_cluster_indices = [i for i in same_cluster_indices if i != index]

        rec_indices = same_cluster_indices[:count]
        return self.songs.iloc[rec_indices].copy()
