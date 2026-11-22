"""Hierarchical song recommendation."""

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F

from src.config import DEFAULT_RECOMMENDATION_COUNT, SONG_FEATURES


class HierarchicalRecommender:
    """Recommend songs from the same bisecting k-means cluster."""

    def __init__(
        self,
        n_clusters: int = 10,
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive.")

        self.n_clusters = n_clusters
        self.recommendation_count = recommendation_count
        self.songs: DataFrame | None = None
        self.song_columns: list[str] = []

    def fit(self, songs: DataFrame) -> "HierarchicalRecommender":
        """Standardize usable song features and assign cluster labels."""
        feature_columns = [name for name in SONG_FEATURES if name in songs.columns]
        clean_songs = songs
        for name in feature_columns:
            value = F.col(name).cast("double")
            is_finite = ~F.isnan(value) & (F.abs(value) < float("inf"))
            clean_songs = clean_songs.withColumn(name, F.when(is_finite, value))

        # Count the songs and the non-missing values for each feature in one pass.
        counts = [F.count("*").alias("song_count")]
        for name in feature_columns:
            counts.append(F.count(name).alias(name))
        observed = clean_songs.agg(*counts).first()
        song_count = observed["song_count"]
        if song_count == 0:
            raise ValueError("Cannot fit hierarchical clustering on no songs.")

        # A feature needs at least one value before Spark can fill in its median.
        feature_columns = [name for name in feature_columns if observed[name] > 0]
        if not feature_columns:
            raise ValueError("No usable numeric song feature values were found.")

        filled_columns = [f"{name}_filled" for name in feature_columns]
        fill_missing = Imputer(
            strategy="median",
            inputCols=feature_columns,
            outputCols=filled_columns,
        )
        make_vector = VectorAssembler(inputCols=filled_columns, outputCol="raw_features")
        # Put features like tempo and loudness on a comparable scale.
        scale_features = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        )
        stages = [fill_missing, make_vector, scale_features]

        cluster_count = min(self.n_clusters, song_count)
        if cluster_count > 1:
            cluster_songs = BisectingKMeans(
                k=cluster_count, seed=42, predictionCol="cluster"
            )
            stages.append(cluster_songs)

        model = Pipeline(stages=stages).fit(clean_songs)
        clustered = model.transform(clean_songs)
        if cluster_count == 1:
            clustered = clustered.withColumn("cluster", F.lit(0))

        self.song_columns = songs.columns
        self.songs = clustered.select(*self.song_columns, "features", "cluster")
        return self

    def _find_seed(self, track_id: str) -> Row:
        if self.songs is None:
            raise RuntimeError("Call fit() before looking up songs.")

        for field in ("song_id", "track_id"):
            if field in self.song_columns:
                seed = self.songs.filter(F.col(field) == track_id).first()
                if seed is not None:
                    return seed

        raise ValueError(f"Song not found in Parquet: {track_id}. Refresh the song export.")

    def find_song(self, track_id: str) -> dict:
        """Find a song and return its cluster assignment."""
        seed = self._find_seed(track_id)
        return {column: seed[column] for column in [*self.song_columns, "cluster"]}

    def get_cluster(self, track_id: str) -> int:
        """Return the cluster label for a song."""
        return self._find_seed(track_id)["cluster"]

    def recommend(self, track_id: str, n: int | None = None) -> pd.DataFrame:
        """Rank other songs in the seed's cluster by Euclidean distance."""
        count = self.recommendation_count if n is None else n
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        seed = self._find_seed(track_id)
        candidates = self.songs.filter(
            (F.col("cluster") == seed["cluster"]) & (F.col("song_id") != seed["song_id"])
        )

        # Build a Spark expression that adds up the squared feature differences.
        # Squared distances sort in the same order as Euclidean distances.
        candidate_features = vector_to_array("features")
        distance_squared = F.lit(0.0)
        for index, seed_value in enumerate(seed["features"]):
            difference = candidate_features[index] - float(seed_value)
            distance_squared = distance_squared + difference * difference

        ranked_songs = candidates.orderBy(distance_squared, "song_id")
        top_songs = ranked_songs.limit(count).select(*self.song_columns)
        return top_songs.toPandas()
