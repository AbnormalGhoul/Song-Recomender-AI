"""Gaussian Mixture Model song recommendation."""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_RECOMMENDATION_COUNT
from src.data_handling.features import find_song_index, prepare_feature_matrix


class GMMRecommender:
    """Recommend songs using Gaussian mixture components."""

    def __init__(
        self,
        n_components: int = 10,
        random_state: int = 42,
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    ):
        if n_components < 1:
            raise ValueError("n_components must be positive.")

        self.n_components = n_components
        self.random_state = random_state
        self.recommendation_count = recommendation_count

        self.model = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
        )
        self.scaler: StandardScaler | None = None
        self.songs: pd.DataFrame | None = None
        self.features: np.ndarray | None = None
        self.feature_columns: list[str] = []
        self.labels: np.ndarray | None = None

    def fit(self, songs: pd.DataFrame) -> "GMMRecommender":
        """Prepare features and fit the Gaussian mixture model."""
        if songs.empty:
            raise ValueError("Cannot fit GMM with an empty dataset.")

        self.songs = songs.reset_index(drop=True).copy()
        self.features, self.scaler, self.feature_columns = (
            prepare_feature_matrix(self.songs)
        )

        if len(self.songs) < self.n_components:
            raise ValueError(
                "The number of songs must be at least n_components."
            )

        self.model.fit(self.features)
        self.labels = self.model.predict(self.features)

        return self

    def find_song(self, track_id: str) -> dict:
        """Return a song record and its assigned component."""
        if self.songs is None:
            raise RuntimeError("Call fit() before finding songs.")

        index = find_song_index(self.songs, track_id)

        if self.labels is None:
            raise NotImplementedError("GMM component labels are not ready.")

        result = self.songs.iloc[index].to_dict()
        result["cluster"] = int(self.labels[index])
        return result

    def predict_cluster(self, features: np.ndarray) -> np.ndarray:
        """Predict mixture component membership for feature vectors."""
        return self.model.predict(features)

    def recommend(
        self,
        track_id: str,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Recommend songs related to the source song's component."""
        if self.songs is None:
            raise RuntimeError("Call fit() before requesting recommendations.")

        count = n if n is not None else self.recommendation_count
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        index = find_song_index(self.songs, track_id)

        if self.labels is None:
            raise NotImplementedError("GMM component labels are not ready.")

        target_cluster = self.labels[index]
        same_cluster_indices = np.where(self.labels == target_cluster)[0]
        same_cluster_indices = [i for i in same_cluster_indices if i != index]

        if same_cluster_indices:
            source_features = self.features[index]
            cluster_features = self.features[same_cluster_indices]
            distances = np.linalg.norm(cluster_features - source_features, axis=1)
            sorted_indices = np.argsort(distances)
            same_cluster_indices = [same_cluster_indices[i] for i in sorted_indices]

        rec_indices = same_cluster_indices[:count]
        return self.songs.iloc[rec_indices].copy()
