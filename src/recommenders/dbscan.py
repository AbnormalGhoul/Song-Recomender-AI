"""Spark-backed DBSCAN song recommendations over MongoDB song data."""

from __future__ import annotations

import pandas as pd
from pyspark import RDD, StorageLevel
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import abs as spark_abs
from pyspark.sql.functions import col, isnan, lit, percentile_approx, when
from pyspark.sql.types import IntegerType, StructField

from src.config import (
    DEFAULT_RECOMMENDATION_COUNT,
    MONGODB_COLLECTION,
    MONGODB_DATABASE,
    MONGODB_URI,
    SONG_FEATURES,
)

class DBSCANRecommender:
    """Distributed DBSCAN with an epsilon grid index and Spark RDD operations.

    Core-point connectivity is computed with distributed neighbor joins and
    label propagation. The grid limits distance checks to nearby cells rather
    than comparing every song with every other song.
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        recommendation_count: int = DEFAULT_RECOMMENDATION_COUNT,
        spark: SparkSession | None = None,
        max_label_iterations: int = 1000,
    ):
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        if recommendation_count < 1:
            raise ValueError("recommendation_count must be positive.")
        if max_label_iterations < 1:
            raise ValueError("max_label_iterations must be positive.")

        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.recommendation_count = int(recommendation_count)
        self.max_label_iterations = int(max_label_iterations)
        self.spark = spark
        self.songs: DataFrame | None = None
        self.labels: RDD | None = None
        self.feature_columns: list[str] = []
        self._scaled_columns: list[str] = []
        self._source_df: DataFrame | None = None

    def _get_spark(self) -> SparkSession:
        if self.spark is None:
            from src.spark.session import get_spark

            self.spark = get_spark("SongRecommender-DBSCAN")
        return self.spark

    def fit_from_mongodb(
        self,
        uri: str = MONGODB_URI,
        database: str = MONGODB_DATABASE,
        collection: str = MONGODB_COLLECTION,
    ) -> "DBSCANRecommender":
        """Read song records through the MongoDB Spark Connector and fit."""
        songs = (
            self._get_spark().read.format("mongodb")
            .option("spark.mongodb.read.connection.uri", uri)
            .option("database", database)
            .option("collection", collection)
            .load()
        )
        return self.fit(songs)

    def fit(self, songs: DataFrame | pd.DataFrame) -> "DBSCANRecommender":
        """Fit DBSCAN to a Spark DataFrame (or a small local pandas frame).

        For production-sized datasets, pass a Spark DataFrame or use
        :meth:`fit_from_mongodb`; passing pandas data requires driver memory
        proportional to the full input size.
        """
        spark = self._get_spark()
        if isinstance(songs, pd.DataFrame):
            if songs.empty:
                raise ValueError("Cannot fit DBSCAN with an empty dataset.")
            source = spark.createDataFrame(songs)
        elif isinstance(songs, DataFrame):
            source = songs
        else:
            raise TypeError("songs must be a Spark or pandas DataFrame.")

        if not source.columns:
            raise ValueError("Cannot fit DBSCAN without song columns.")

        self._source_df = source.drop("_id") if "_id" in source.columns else source
        self.feature_columns = [
            name for name in SONG_FEATURES if name in self._source_df.columns
        ]
        if not self.feature_columns:
            raise ValueError("No usable numeric song features were found.")

        # Cast to doubles, convert invalid and non-finite values to null, then
        # impute medians and standardize using distributed aggregate actions.
        casted = self._source_df.select(
            "*", *[
                col(name).cast("double").alias(f"__numeric_{name}")
                for name in self.feature_columns
            ]
        )
        numeric_names = [f"__numeric_{name}" for name in self.feature_columns]
        # Spark casts NaN to a value rather than null; replace it explicitly.
        for name in numeric_names:
            casted = casted.withColumn(
                name,
                when(
                    col(name).isNull()
                    | isnan(col(name))
                    | (spark_abs(col(name)) == lit(float("inf"))),
                    None,
                ).otherwise(col(name)),
            )

        medians = casted.agg(*[
            percentile_approx(col(name), 0.5, 10000).alias(name)
            for name in numeric_names
        ]).first()
        valid_features = [
            name for name in self.feature_columns
            if medians[f"__numeric_{name}"] is not None
        ]
        if not valid_features:
            raise ValueError("No usable numeric feature values were found.")

        self.feature_columns = valid_features
        numeric_names = [f"__numeric_{name}" for name in valid_features]
        imputed = casted
        for name in numeric_names:
            imputed = imputed.withColumn(
                name, col(name).cast("double")
            ).na.fill({name: float(medians[name])})

        aggregates = []
        for name in numeric_names:
            aggregates.extend([
                col(name).mean().alias(f"__mean_{name}"),
                col(name).stddev_pop().alias(f"__std_{name}"),
            ])
        stats = imputed.agg(*aggregates).first()
        self._scaled_columns = [f"__scaled_{name}" for name in valid_features]
        scaled = imputed
        for name, numeric_name, scaled_name in zip(
            valid_features, numeric_names, self._scaled_columns
        ):
            mean = float(stats[f"__mean_{numeric_name}"] or 0.0)
            std = float(stats[f"__std_{numeric_name}"] or 0.0)
            divisor = std if std > 0 else 1.0
            scaled = scaled.withColumn(
                scaled_name, (col(numeric_name) - lit(mean)) / lit(divisor)
            )

        keep_columns = [
            name for name in self._source_df.columns
            if name in scaled.columns and not name.startswith("__numeric_")
        ]
        working = scaled.select(*keep_columns, *self._scaled_columns)
        self._cluster(working)
        return self

    def _cluster(self, working: DataFrame) -> None:
        dimensions = len(self._scaled_columns)
        eps_squared = self.eps * self.eps
        eps = self.eps
        minimum_samples = self.min_samples
        scaled_columns = self._scaled_columns
        def to_point(pair: tuple[Row, int]) -> tuple[int, tuple[tuple[float, ...], dict]]:
            row, point_id = pair
            record = row.asDict(recursive=True)
            vector = tuple(float(record[name]) for name in scaled_columns)
            return int(point_id), (vector, record)

        points = working.rdd.zipWithIndex().map(to_point).persist(
            StorageLevel.MEMORY_AND_DISK
        )

        def cell(vector: tuple[float, ...]) -> tuple[int, ...]:
            return tuple(int(value // eps) for value in vector)

        cells = points.map(lambda item: (cell(item[1][0]), item)).groupByKey().mapValues(
            list
        ).persist(StorageLevel.MEMORY_AND_DISK)

        def candidate_cells(item: tuple[int, tuple[tuple[float, ...], dict]]):
            point_id, (vector, _) = item
            base = cell(vector)

            def visit(dimension: int, key: tuple[int, ...], min_distance: float):
                if dimension == dimensions:
                    yield key, (point_id, vector)
                    return
                value = vector[dimension]
                for delta in (-1, 0, 1):
                    index = base[dimension] + delta
                    low, high = index * eps, (index + 1) * eps
                    gap = low - value if value < low else value - high if value > high else 0.0
                    distance = min_distance + gap * gap
                    if distance <= eps_squared:
                        yield from visit(dimension + 1, key + (index,), distance)

            yield from visit(0, (), 0.0)

        def nearby_pairs(item):
            _, ((point_id, vector), members) = item
            for other_id, (other_vector, _) in members:
                distance_squared = sum(
                    (left - right) ** 2 for left, right in zip(vector, other_vector)
                )
                if distance_squared <= eps_squared:
                    yield point_id, other_id

        neighbors = (
            points.flatMap(candidate_cells)
            .join(cells)
            .flatMap(nearby_pairs)
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        core_ids = (
            neighbors.map(lambda pair: (pair[0], 1))
            .reduceByKey(lambda left, right: left + right)
            .filter(lambda item: item[1] >= minimum_samples)
            .keys()
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        # A connected component of core points is one DBSCAN cluster. Iterative
        # minimum-label propagation uses Spark joins/reductions across workers.
        core_markers = core_ids.map(lambda point_id: (point_id, True)).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        core_edges = (
            neighbors
            .map(lambda pair: (pair[1], pair[0]))
            .join(core_markers)
            .map(lambda item: (item[1][0], item[0]))
            .join(core_markers)
            .map(lambda item: (item[0], item[1][0]))
            .distinct()
            .persist(StorageLevel.MEMORY_AND_DISK)
        )

        labels = core_ids.map(lambda point_id: (point_id, point_id)).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        converged = False
        for _ in range(self.max_label_iterations):
            propagated = (
                core_edges
                .map(lambda edge: (edge[1], edge[0]))
                .join(labels)
                .map(lambda item: (item[1][0], item[1][1]))
                .union(labels)
                .reduceByKey(min)
                .persist(StorageLevel.MEMORY_AND_DISK)
            )
            changed = labels.join(propagated).filter(
                lambda item: item[1][0] != item[1][1]
            ).take(1)
            labels.unpersist()
            if not changed:
                labels = propagated
                converged = True
                break
            labels = propagated

        if not converged:
            raise RuntimeError(
                "DBSCAN cluster labeling did not converge within "
                f"{self.max_label_iterations} Spark iterations."
            )

        border_labels = (
            neighbors
            .map(lambda pair: (pair[1], pair[0]))
            .join(labels)
            .map(lambda item: (item[1][0], item[1][1]))
            .reduceByKey(min)
            .leftOuterJoin(core_markers)
            .filter(lambda item: item[1][1] is None)
            .mapValues(lambda item: item[0])
        )
        cluster_labels = labels.union(border_labels).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        schema = working.schema.add(
            StructField("cluster", IntegerType(), nullable=False)
        )
        clustered_rows = (
            cluster_labels.join(points)
            .map(lambda item: Row(**item[1][1][1], cluster=int(item[1][0])))
        )
        self.songs = self._get_spark().createDataFrame(
            clustered_rows, schema=schema
        ).persist(StorageLevel.MEMORY_AND_DISK)
        self.songs.count()
        self.labels = cluster_labels

        # Keep the expensive indexed data cached for recommendations. The
        # original point records are represented in ``self.songs`` instead.
        points.unpersist()
        cells.unpersist()
        neighbors.unpersist()
        core_ids.unpersist()
        core_markers.unpersist()
        core_edges.unpersist()

    def find_song(self, track_id: str) -> dict:
        """Find a song and its DBSCAN cluster assignment."""
        if self.songs is None:
            raise RuntimeError("Call fit() before finding songs.")
        if "track_id" not in self.songs.columns and "song_id" not in self.songs.columns:
            raise ValueError("Song records need a track_id or song_id column.")

        id_column = "track_id" if "track_id" in self.songs.columns else "song_id"
        row = self.songs.filter(col(id_column).cast("string") == str(track_id)).limit(1).first()
        if row is None and id_column == "track_id" and "song_id" in self.songs.columns:
            row = self.songs.filter(col("song_id").cast("string") == str(track_id)).limit(1).first()
        if row is None:
            raise ValueError(f"Song not found: {track_id}")
        return row.asDict(recursive=True)

    def get_cluster(self, track_id: str) -> int:
        """Return the cluster label for a specific song; -1 means noise."""
        return int(self.find_song(track_id)["cluster"])

    def recommend(self, track_id: str, n: int | None = None) -> pd.DataFrame:
        """Return the nearest other songs in the target song's DBSCAN cluster."""
        if self.songs is None:
            raise RuntimeError("Call fit() before requesting recommendations.")
        count = n if n is not None else self.recommendation_count
        if count < 1:
            raise ValueError("Recommendation count must be positive.")

        target = self.find_song(track_id)
        target_cluster = int(target["cluster"])
        if target_cluster == -1:
            public_columns = [
                name for name in self.songs.columns
                if name not in self._scaled_columns
            ]
            return pd.DataFrame(columns=public_columns)

        distance = None
        for scaled_name in self._scaled_columns:
            target_value = float(target.get(scaled_name, 0.0))
            term = (col(scaled_name) - lit(target_value)) ** 2
            distance = term if distance is None else distance + term

        recs = (
            self.songs.filter(col("cluster") == target_cluster)
        )
        for field in ("song_id", "track_id"):
            if field in self.songs.columns:
                recs = recs.filter(col(field).cast("string") != str(track_id))
        if distance is not None:
            recs = recs.withColumn("__distance", distance).orderBy("__distance")
        public_columns = [
            name for name in self.songs.columns
            if name not in self._scaled_columns
        ]
        return pd.DataFrame.from_records(
            [
                {name: values[name] for name in public_columns}
                for values in (
                    row.asDict(recursive=True)
                    for row in recs.limit(count).collect()
                )
            ]
        )
