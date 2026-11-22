"""Command-line entry point for Spark DBSCAN recommendations."""

import argparse
import sys
import time

from src.database.database import SongDatabase
from src.recommenders.dbscan import DBSCANRecommender
from src.spark.session import get_spark


def main() -> int:
    parser = argparse.ArgumentParser(description="Spark DBSCAN song recommender")
    parser.add_argument("--title", required=True, help="Song title")
    parser.add_argument("--artist", required=True, help="Song artist")
    parser.add_argument("--eps", type=float, default=0.5, help="DBSCAN radius")
    parser.add_argument(
        "--min-samples", type=int, default=5,
        help="Minimum songs in an epsilon neighborhood",
    )
    parser.add_argument(
        "--recommendations", type=int, default=10,
        help="Maximum recommendations to return",
    )
    args = parser.parse_args()

    try:
        with SongDatabase() as db:
            song = db.find_by_title_artist(args.title, args.artist)
            if not song:
                print(
                    f"Error: Song '{args.title}' by '{args.artist}' "
                    "not found in the database.",
                    file=sys.stderr,
                )
                return 1
            track_id = song.get("track_id") or song.get("song_id")

        spark = get_spark("SongRecommender-DBSCAN")
        recommender = DBSCANRecommender(
            eps=args.eps,
            min_samples=args.min_samples,
            recommendation_count=args.recommendations,
            spark=spark,
        )
        started_at = time.perf_counter()
        recommender.fit_from_mongodb()
        fit_seconds = time.perf_counter() - started_at
        recommendations = recommender.recommend(track_id)
        cluster_counts = {
            int(row["cluster"]): int(row["count"])
            for row in recommender.songs.groupBy("cluster").count().collect()
        }

        print(
            f"Found song: {song.get('title', args.title)} by "
            f"{song.get('artist', args.artist)} (ID: {track_id})"
        )
        print(f"DBSCAN cluster: {recommender.get_cluster(track_id)}")
        print(
            f"Fit time: {fit_seconds:.2f}s; "
            f"songs: {sum(cluster_counts.values())}; "
            f"clusters: {sum(1 for cluster in cluster_counts if cluster != -1)}; "
            f"noise: {cluster_counts.get(-1, 0)}"
        )
        if recommendations.empty:
            print("No recommendations found (the song may be noise).")
        else:
            print("--- DBSCAN Recommendations ---")
            for index, row in recommendations.iterrows():
                print(
                    f"{index + 1}. {row.get('title', 'Unknown')} by "
                    f"{row.get('artist', row.get('artist_name', 'Unknown'))}"
                )
        return 0
    except Exception as error:
        print(f"Failed to generate DBSCAN recommendations: {error}", file=sys.stderr)
        return 1
    finally:
        try:
            from pyspark.sql import SparkSession

            active_spark = SparkSession.getActiveSession()
            if active_spark is not None:
                active_spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
