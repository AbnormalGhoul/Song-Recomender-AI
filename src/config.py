"""Central configuration for the Song Recommender AI project."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "song_recommender")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "songs")

# Ingestion and recommendation defaults.
BATCH_SIZE = 500
DEFAULT_RECOMMENDATION_COUNT = 10

# Features used for clustering and similarity comparisons.
SONG_FEATURES = [
    "duration",
    "tempo",
    "loudness",
    "key",
    "mode",
    "time_signature",
    "year",
]

# MongoDB indexes used to avoid duplicate songs and speed up lookups.
UNIQUE_SONG_FIELD = "track_id"

# Spark / distributed-processing settings.
PARQUET_DIR = Path(os.getenv("PARQUET_DIR", str(DATA_DIR / "parquet" / "songs")))
RESULTS_DIR = PROJECT_ROOT / "results"
SPARK_MASTER = os.getenv("SPARK_MASTER")  # e.g. "local[*]" or "spark://host:7077"
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "64"))
MONGO_SPARK_PACKAGE = os.getenv(
    "MONGO_SPARK_PACKAGE",
    "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0",
)