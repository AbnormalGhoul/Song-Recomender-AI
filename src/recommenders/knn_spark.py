"""Distributed K-nearest-neighbor song recommendation with Spark MLlib."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    BucketedRandomProjectionLSH,
    Imputer,
    Normalizer,
    StandardScaler,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.config import DEFAULT_RECOMMENDATION_COUNT, SONG_FEATURES

OUTPUT_COLUMNS = ["song_id", "track_id", "title", "artist"]


class SparkKNNRecommender:
    """Approximate cosine KNN over all songs using Spark MLlib LSH."""

    def __init__(
        self,
        feature_columns: Sequence[str] | None = None,
        bucket_length: float = 0.5,
        num_hash_tables: int = 5,
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
        seed: int = 42,
    ):
        if bucket_length <= 0:
            raise ValueError("bucket_length must be positive.")
        if num_hash_tables < 1:
            raise ValueError("num_hash_tables must be at least 1.")

        self.requested_features = list(feature_columns or SONG_FEATURES)
        self.bucket_length = bucket_length
        self.num_hash_tables = num_hash_tables
        self.recommendation_count = recommendation_count
        self.seed = seed

        self.feature_columns: list[str] = []
        self.pipeline_model: PipelineModel | None = None
        self.indexed: DataFrame | None = None

    # fit

    def _build_pipeline(self, columns: list[str]) -> Pipeline:
        imputed = [f"{c}_imputed" for c in columns]
        return Pipeline(stages=[
            Imputer(strategy="median", inputCols=columns, outputCols=imputed),
            VectorAssembler(inputCols=imputed, outputCol="raw_features"),
            StandardScaler(
                inputCol="raw_features",
                outputCol="scaled_features",
                withMean=True,
                withStd=True,
            ),
            Normalizer(inputCol="scaled_features", outputCol="features", p=2.0),
            BucketedRandomProjectionLSH(
                inputCol="features",
                outputCol="hashes",
                bucketLength=self.bucket_length,
                numHashTables=self.num_hash_tables,
                seed=self.seed,
            ),
        ])

    def fit(self, songs: DataFrame) -> "SparkKNNRecommender":
        """Fit the feature pipeline and LSH model, then cache hashed songs."""
        self.feature_columns = [c for c in self.requested_features if c in songs.columns]
        if not self.feature_columns:
            raise ValueError("No usable numeric song features were found.")

        songs = songs.select(
            *[F.col(c) for c in OUTPUT_COLUMNS if c in songs.columns],
            *[F.col(c).cast("double").alias(c) for c in self.feature_columns],
        )

        self.pipeline_model = self._build_pipeline(self.feature_columns).fit(songs)
        self._index(songs)
        return self

    def _index(self, songs: DataFrame) -> None:
        """Transform all songs once; keep only what queries need, cached."""
        if self.indexed is not None:
            self.indexed.unpersist()

        self.indexed = (
            self.pipeline_model.transform(songs)
            .select(*[c for c in OUTPUT_COLUMNS if c in songs.columns], "features", "hashes")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        self.indexed.count()  # materialize the cache

    @property
    def lsh_model(self):
        return self._require_model().stages[-1]

    def _require_model(self) -> PipelineModel:
        if self.pipeline_model is None or self.indexed is None:
            raise RuntimeError("Call fit() (or load()) before requesting recommendations.")
        return self.pipeline_model

    # persistence

    def save(self, path: Path | str) -> None:
        """Save the fitted pipeline (scaler stats + LSH hash functions)."""
        self._require_model().write().overwrite().save(str(path))

    def load(self, path: Path | str, songs: DataFrame) -> "SparkKNNRecommender":
        """Load a saved pipeline and hash the given songs with it."""
        self.pipeline_model = PipelineModel.load(str(path))
        imputer = self.pipeline_model.stages[0]
        self.feature_columns = list(imputer.getInputCols())
        songs = songs.select(
            *[F.col(c) for c in OUTPUT_COLUMNS if c in songs.columns],
            *[F.col(c).cast("double").alias(c) for c in self.feature_columns],
        )
        self._index(songs)
        return self

    # queries

    def find_song(self, song_id: str) -> dict:
        """Return the indexed row for a song_id or track_id."""
        self._require_model()
        condition = F.col("song_id") == str(song_id)
        if "track_id" in self.indexed.columns:
            condition = condition | (F.col("track_id") == str(song_id))

        row = self.indexed.filter(condition).limit(1).collect()
        if not row:
            raise ValueError(f"Song not found: {song_id}")
        return row[0].asDict()

    @staticmethod
    def _with_similarity(df: DataFrame) -> DataFrame:
        # Unit vectors: cos = 1 - d^2 / 2
        return df.withColumn(
            "cosine_similarity", F.lit(1.0) - F.pow(F.col("distance"), 2) / 2.0
        )

    def recommend(self, track_id: str, n: int | None = None) -> pd.DataFrame:
        """Approximate top-n most similar songs (LSH), excluding the seed."""
        count = n if n is not None else self.recommendation_count
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        seed = self.find_song(track_id)
        neighbors = self.lsh_model.approxNearestNeighbors(
            self.indexed, seed["features"], count + 1, distCol="distance"
        )

        recs = (
            self._with_similarity(neighbors)
            .filter(F.col("song_id") != seed["song_id"])
            .orderBy("distance")
            .limit(count)
            .select(*[c for c in OUTPUT_COLUMNS if c in self.indexed.columns],
                    "distance", "cosine_similarity")
        )
        return recs.toPandas()

    def recommend_exact(self, track_id: str, n: int | None = None) -> pd.DataFrame:
        """Exact top-n by a full distributed scan. Used as ground truth."""
        count = n if n is not None else self.recommendation_count
        seed = self.find_song(track_id)

        seed_array = F.array(*[F.lit(float(x)) for x in seed["features"].toArray()])
        dot = F.aggregate(
            F.zip_with(vector_to_array("features"), seed_array, lambda a, b: a * b),
            F.lit(0.0),
            lambda acc, x: acc + x,
        )

        recs = (
            self.indexed.withColumn("cosine_similarity", dot)
            .withColumn(
                "distance",
                F.sqrt(F.greatest(F.lit(0.0), 2.0 - 2.0 * F.col("cosine_similarity"))),
            )
            .filter(F.col("song_id") != seed["song_id"])
            .orderBy(F.col("distance").asc())
            .limit(count)
            .select(*[c for c in OUTPUT_COLUMNS if c in self.indexed.columns],
                    "distance", "cosine_similarity")
        )
        return recs.toPandas()

    # offline batch

    def precompute_all(self, k: int | None = None, min_similarity: float = 0.95) -> DataFrame:
        """Top-k neighbors for every song via an LSH similarity self join.

        Only pairs whose cosine similarity is at least min_similarity are
        considered (distance threshold sqrt(2 * (1 - min_similarity))), so
        a song in a sparse region may get fewer than k neighbors.
        """
        self._require_model()
        count = k if k is not None else self.recommendation_count
        threshold = (2.0 * (1.0 - min_similarity)) ** 0.5

        pairs = self.lsh_model.approxSimilarityJoin(
            self.indexed, self.indexed, threshold, distCol="distance"
        ).select(
            F.col("datasetA.song_id").alias("song_id"),
            F.col("datasetB.song_id").alias("recommended_song_id"),
            F.col("datasetB.title").alias("recommended_title"),
            F.col("datasetB.artist").alias("recommended_artist"),
            "distance",
        ).filter(F.col("song_id") != F.col("recommended_song_id"))

        rank = Window.partitionBy("song_id").orderBy(F.col("distance").asc())
        return (
            self._with_similarity(pairs)
            .withColumn("rank", F.row_number().over(rank))
            .filter(F.col("rank") <= count)
        )