"""Export the MongoDB songs collection to cleaned, columnar Parquet files.

MongoDB stays the system of record; Parquet is the analytics copy that the
Spark algorithms read. Parquet is columnar and compressed, so a job that
needs 7 feature columns reads only those columns instead of whole
documents, and the files split naturally across Spark executors/HDFS
blocks.

Usage:
    python -m src.spark.export_parquet
    python -m src.spark.export_parquet --out data/parquet/songs --partitions 16
"""

import argparse
import time
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import PARQUET_DIR
from src.spark.session import get_spark, read_mongo_songs

ID_COLUMNS = ["song_id", "track_id", "title", "artist", "release", "artist_id"]

NUMERIC_COLUMNS = [
    "duration",
    "end_of_fade_in",
    "loudness",
    "tempo",
    "key",
    "key_confidence",
    "mode",
    "mode_confidence",
    "time_signature",
    "time_signature_confidence",
    "artist_familiarity",
    "artist_hotness",
    "song_hotness",
    "year",
]


def clean_songs(df: DataFrame) -> DataFrame:
    """Select, type, and clean the fields used by the recommenders."""
    present = set(df.columns)

    selected = [F.col(c).cast("string").alias(c) for c in ID_COLUMNS if c in present]
    selected += [F.col(c).cast("double").alias(c) for c in NUMERIC_COLUMNS if c in present]
    df = df.select(*selected)

    # NaN -> null so Spark's Imputer and aggregates treat them as missing.
    for column in NUMERIC_COLUMNS:
        if column in present:
            df = df.withColumn(
                column,
                F.when(F.isnan(F.col(column)), None).otherwise(F.col(column)),
            )

    # The MSD stores an unknown year as 0, not as missing.
    if "year" in present:
        df = df.withColumn(
            "year", F.when(F.col("year") > 0, F.col("year")).otherwise(None)
        )

    return (
        df.filter(F.col("song_id").isNotNull() & (F.trim("song_id") != ""))
        .dropDuplicates(["song_id"])
    )


def export(out: Path, partitions: int) -> dict:
    """Read MongoDB, clean, and write Parquet. Returns summary stats."""
    spark = get_spark("SongRecommender-ExportParquet")

    started = time.perf_counter()
    raw = read_mongo_songs(spark)
    songs = clean_songs(raw).repartition(partitions)
    songs.write.mode("overwrite").parquet(str(out))
    elapsed = time.perf_counter() - started

    written = spark.read.parquet(str(out))
    stats = {
        "rows": written.count(),
        "columns": len(written.columns),
        "files": len(list(Path(out).glob("*.parquet"))),
        "bytes": sum(p.stat().st_size for p in Path(out).glob("*.parquet")),
        "seconds": round(elapsed, 2),
    }

    print("Parquet schema:")
    written.printSchema()
    print("Sample rows:")
    sample_columns = [
        c for c in ("song_id", "title", "artist", "tempo", "loudness", "year")
        if c in written.columns
    ]
    written.select(*sample_columns).show(5, truncate=40)
    print(
        f"Wrote {stats['rows']:,} songs to {out} "
        f"({stats['files']} files, {stats['bytes'] / 1e6:.1f} MB) "
        f"in {stats['seconds']} s"
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MongoDB songs to Parquet.")
    parser.add_argument("--out", type=Path, default=PARQUET_DIR)
    parser.add_argument("--partitions", type=int, default=8)
    args = parser.parse_args()
    export(args.out, args.partitions)


if __name__ == "__main__":
    main()