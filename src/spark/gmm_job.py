# heres an example command to run it:
# python -m src.spark.gmm_job --title "Song Title" --artist "Artist"

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import DEFAULT_RECOMMENDATION_COUNT, PARQUET_DIR, SPARK_MASTER, SPARK_SHUFFLE_PARTITIONS
from src.recommenders.gmm_spark import SparkGMMRecommender

def _seed_id(songs: DataFrame, args: argparse.Namespace) -> str:
    if args.song_id is not None:
        row = songs.filter(F.col("song_id") == args.song_id).select("song_id").first()
        if row is None:
            raise SystemExit(f"Song ID '{args.song_id}' not found.")
        return row.song_id

    matches = (
        songs.filter(
            (F.lower(F.trim("title")) == args.title.strip().lower())
            & (F.lower(F.trim("artist")) == args.artist.strip().lower())
        )
        .select("song_id")
        .distinct()
        .limit(5)
        .collect()
    )
    if not matches:
        raise SystemExit(f"Song '{args.title}' by '{args.artist}' not found.")
    if len(matches) > 1:
        ids = ", ".join(row.song_id for row in matches)
        raise SystemExit(f"More than one song matched. Run again with --song-id. Matching IDs: {ids}")
    return matches[0].song_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend songs using Spark GMM.")
    seed = parser.add_mutually_exclusive_group(required=True)
    seed.add_argument("--song-id", help="ID of the seed song")
    seed.add_argument("--title", help="Title of the seed song, used with --artist")
    parser.add_argument("--artist", help="Artist of the seed song, used with --title")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DIR, help="Song catalog path")
    parser.add_argument("--components", type=int, default=10, help="Number of groups to learn")
    parser.add_argument(
        "--k", type=int, default=DEFAULT_RECOMMENDATION_COUNT,
        help="Maximum number of recommendations",
    )
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    args = parser.parse_args()

    if (args.title is None) != (args.artist is None):
        parser.error("Use --title and --artist together, or use --song-id by itself.")
    if args.components < 2 or args.k < 1:
        parser.error("--components must be at least 2 and --k must be positive.")
    if not args.parquet.exists():
        raise SystemExit(f"No Parquet data at {args.parquet}. Run python -m src.spark.export_parquet.")

    builder = SparkSession.builder.appName("SongRecommender-GMM").config(
        "spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS
    )
    if SPARK_MASTER:
        builder = builder.master(SPARK_MASTER)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        songs = spark.read.parquet(str(args.parquet))
        song_id = _seed_id(songs, args)
        model = SparkGMMRecommender(n_components=args.components, seed=args.seed)
        model.fit(songs)
        recommendations = model.recommend(song_id, n=args.k).collect()

        if not recommendations:
            print("No other songs in the seed song's group.")
        for rank, row in enumerate(recommendations, 1):
            song = row.asDict()
            title = song.get("title") or song["song_id"]
            artist = song.get("artist") or "Unknown artist"
            print(f"{rank:2}. {title} by {artist} (distance={song['distance']:.4f})")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
