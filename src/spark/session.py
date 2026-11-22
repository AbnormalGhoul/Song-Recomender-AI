"""Create a SparkSession wired to the project's MongoDB instance."""

from pyspark.sql import DataFrame, SparkSession

from src.config import (
    MONGO_SPARK_PACKAGE,
    MONGODB_COLLECTION,
    MONGODB_DATABASE,
    MONGODB_URI,
    SPARK_MASTER,
    SPARK_SHUFFLE_PARTITIONS,
)


def get_spark(app_name: str = "SongRecommender", *, with_mongo: bool = True) -> SparkSession:
    """Return a SparkSession, optionally with the MongoDB Spark Connector configured.

    Runs in local mode unless SPARK_MASTER points at a cluster.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.parquet.compression.codec", "snappy")
    )

    if with_mongo:
        builder = (
            builder.config("spark.jars.packages", MONGO_SPARK_PACKAGE)
            .config("spark.mongodb.read.connection.uri", MONGODB_URI)
            .config("spark.mongodb.write.connection.uri", MONGODB_URI)
        )

    if SPARK_MASTER:
        builder = builder.master(SPARK_MASTER)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_mongo_songs(
    spark: SparkSession,
    database: str = MONGODB_DATABASE,
    collection: str = MONGODB_COLLECTION,
) -> DataFrame:
    """Load the songs collection from MongoDB as a Spark DataFrame."""
    return (
        spark.read.format("mongodb")
        .option("database", database)
        .option("collection", collection)
        .load()
    )
