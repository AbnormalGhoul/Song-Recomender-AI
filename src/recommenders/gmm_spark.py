from pyspark.ml import Pipeline
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import DEFAULT_RECOMMENDATION_COUNT


class SparkGMMRecommender:
    def __init__(self, n_components: int = 10, seed: int = 42):
        if n_components < 2:
            raise ValueError("Spark GMM needs at least two components.")
        
        self.n_components = n_components
        self.seed = seed
        self.scored_songs: DataFrame | None = None

    def fit(self, songs: DataFrame) -> "SparkGMMRecommender":
        features = [name for name in ("duration", "tempo", "loudness") if name in songs.columns]
        meta = [name for name in ("song_id", "title", "artist") if name in songs.columns]
        clean_features = []

        for name in features:
            value = F.col(name).cast("double")
            clean_features.append(F.when(F.abs(value) < float("inf"), value).alias(name))
        songs = songs.select(*meta, *clean_features)

        stats = songs.agg(
            *[F.min(name).alias(f"{name}_min") for name in features],
            *[F.max(name).alias(f"{name}_max") for name in features],
        ).first()

        features = [name for name in features if stats[f"{name}_min"] != stats[f"{name}_max"]]

        imputed = [f"{name}_imputed" for name in features]
        pipeline = Pipeline(stages=[
            Imputer(strategy="median", inputCols=features, outputCols=imputed),
            VectorAssembler(inputCols=imputed, outputCol="raw_features"),
            StandardScaler(inputCol="raw_features", outputCol="features", withMean=True),
            GaussianMixture(k=self.n_components, seed=self.seed, predictionCol="cluster"),
        ])

        model = pipeline.fit(songs)
        self.scored_songs = model.transform(songs).select(
            *meta, "features", "cluster", "probability"
        )
        return self

    def recommend(self, song_id: str, n: int = DEFAULT_RECOMMENDATION_COUNT) -> DataFrame:
        songs = self.scored_songs

        seed = songs.filter(F.col("song_id") == song_id).select("features", "cluster").first()

        seed_features = F.lit([float(value) for value in seed.features])
        squared_gaps = F.zip_with(
            vector_to_array("features"), seed_features, lambda a, b: (a - b) * (a - b)
        )
        distance = F.sqrt(F.aggregate(squared_gaps, F.lit(0.0), lambda total, gap: total + gap))
        return (songs.filter((F.col("cluster") == seed.cluster) & (F.col("song_id") != song_id))
            .withColumn("distance", distance)
            .orderBy("distance", "song_id")
            .limit(n)
            .drop("features", "cluster", "probability")
        )
