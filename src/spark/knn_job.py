"""Command-line jobs for the Spark KNN recommender,

    # 1) one time: MongoDB -> Parquet
    python -m src.spark.export_parquet

    # 2) recommendations for a seed song
    python -m src.spark.knn_job recommend --title "Song Title" --artist "Artist"

    # 3) metrics for the Milestone 3 report (writes results/knn_spark_*.{json,md,csv})
    python -m src.spark.knn_job evaluate --queries 20

    # 4) offline batch: top k for every song, saved to Parquet
    python -m src.spark.knn_job precompute --min-similarity 0.95 [--to-mongo]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from pyspark.sql import functions as F

from src.config import (
    DEFAULT_RECOMMENDATION_COUNT,
    MODEL_DIR,
    MONGODB_DATABASE,
    PARQUET_DIR,
    RESULTS_DIR,
)
from src.recommenders.knn_spark import SparkKNNRecommender
from src.spark.session import get_spark

MODEL_PATH = MODEL_DIR / "knn_spark"


def _timed(fn, *args, **kwargs):
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - started


def _load_songs(spark, parquet: Path):
    if not Path(parquet).exists():
        raise SystemExit(
            f"No Parquet data at {parquet}. Run: python -m src.spark.export_parquet"
        )
    return spark.read.parquet(str(parquet))


def _fit(args, spark):
    songs, load_s = _timed(_load_songs, spark, args.parquet)
    songs = songs.cache()
    count = songs.count()

    model = SparkKNNRecommender(
        bucket_length=args.bucket_length,
        num_hash_tables=args.hash_tables,
    )
    _, fit_s = _timed(model.fit, songs)
    model.save(MODEL_PATH)
    return model, songs, count, load_s, fit_s


def _seed_id(model, songs, title, artist) -> str:
    normalize = lambda v: " ".join(v.casefold().split())
    rows = (
        songs.filter(
            (F.lower(F.trim("title")) == normalize(title))
            & (F.lower(F.trim("artist")) == normalize(artist))
        )
        .select("song_id")
        .limit(1)
        .collect()
    )
    if not rows:
        raise SystemExit(f"Song '{title}' by '{artist}' not found.")
    return rows[0]["song_id"]


def _print_recs(df, label):
    print(f"\n--- {label} ---")
    for i, row in enumerate(df.itertuples(index=False), 1):
        print(f"{i:2}. {row.title} by {row.artist}  (cosine={row.cosine_similarity:.4f})")


# commands

def cmd_recommend(args) -> None:
    spark = get_spark("SongRecommender-KNN")
    model, songs, count, _, fit_s = _fit(args, spark)
    seed = _seed_id(model, songs, args.title, args.artist)
    recs, q_s = _timed(model.recommend, seed, args.k)
    print(f"Indexed {count:,} songs (fit {fit_s:.2f} s). Query took {q_s:.2f} s.")
    _print_recs(recs, f"Spark KNN (LSH) recommendations for '{args.title}' by {args.artist}")


def cmd_evaluate(args) -> None:
    spark = get_spark("SongRecommender-KNN-Evaluate")
    model, songs, count, load_s, fit_s = _fit(args, spark)
    k = args.k

    seeds = [
        r["song_id"]
        for r in model.indexed.select("song_id")
        .orderBy(F.rand(seed=7))
        .limit(args.queries)
        .collect()
    ]

    approx_times, exact_times, recalls, sims = [], [], [], []
    for song_id in seeds:
        approx, a_s = _timed(model.recommend, song_id, k)
        exact, e_s = _timed(model.recommend_exact, song_id, k)
        approx_times.append(a_s)
        exact_times.append(e_s)
        overlap = set(approx["song_id"]) & set(exact["song_id"])
        recalls.append(len(overlap) / k)
        if len(approx):
            sims.append(float(approx["cosine_similarity"].mean()))

    # Sample output for the report
    sample_id = (
        _seed_id(model, songs, args.title, args.artist)
        if args.title and args.artist
        else seeds[0]
    )
    sample_seed = model.find_song(sample_id)
    sample = model.recommend(sample_id, k)

    def p95(values):
        return sorted(values)[max(0, int(round(0.95 * len(values))) - 1)]

    metrics = {
        "songs_indexed": count,
        "features": model.feature_columns,
        "spark_master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "parquet_partitions": songs.rdd.getNumPartitions(),
        "bucket_length": args.bucket_length,
        "num_hash_tables": args.hash_tables,
        "k": k,
        "queries": len(seeds),
        "parquet_load_s": round(load_s, 3),
        "fit_s": round(fit_s, 3),
        "lsh_query_mean_s": round(statistics.mean(approx_times), 3),
        "lsh_query_median_s": round(statistics.median(approx_times), 3),
        "lsh_query_p95_s": round(p95(approx_times), 3),
        "exact_query_mean_s": round(statistics.mean(exact_times), 3),
        "recall_at_k_mean": round(statistics.mean(recalls), 4),
        "recall_at_k_min": round(min(recalls), 4),
        "mean_cosine_similarity_of_recs": round(statistics.mean(sims), 4) if sims else None,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"b{args.bucket_length}_t{args.hash_tables}"
    (RESULTS_DIR / f"knn_spark_metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    sample.to_csv(RESULTS_DIR / f"knn_spark_sample_{tag}.csv", index=False)

    md = [
        "| Metric | Value |",
        "|---|---|",
        f"| Songs indexed | {count:,} |",
        f"| Features | {', '.join(model.feature_columns)} |",
        f"| Spark master / parallelism | {metrics['spark_master']} / {metrics['default_parallelism']} |",
        f"| LSH params | bucketLength={args.bucket_length}, numHashTables={args.hash_tables} |",
        f"| Parquet load time | {metrics['parquet_load_s']} s |",
        f"| Fit time (impute + scale + normalize + LSH, cached) | {metrics['fit_s']} s |",
        f"| LSH query latency, mean / median / p95 ({len(seeds)} queries) | "
        f"{metrics['lsh_query_mean_s']} / {metrics['lsh_query_median_s']} / {metrics['lsh_query_p95_s']} s |",
        f"| Exact (full scan) query latency, mean | {metrics['exact_query_mean_s']} s |",
        f"| Recall@{k} of LSH vs exact, mean (min) | {metrics['recall_at_k_mean']} ({metrics['recall_at_k_min']}) |",
        f"| Mean cosine similarity of recommendations | {metrics['mean_cosine_similarity_of_recs']} |",
        "",
        f"**Sample output** - seed: *{sample_seed.get('title')}* by {sample_seed.get('artist')}",
        "",
        "| # | Title | Artist | Cosine similarity |",
        "|---|---|---|---|",
    ]
    md += [
        f"| {i} | {r.title} | {r.artist} | {r.cosine_similarity:.4f} |"
        for i, r in enumerate(sample.itertuples(index=False), 1)
    ]
    md_path = RESULTS_DIR / f"knn_spark_results_{tag}.md"
    md_path.write_text("\n".join(md) + "\n")

    print(json.dumps(metrics, indent=2))
    _print_recs(sample, f"Sample: '{sample_seed.get('title')}' by {sample_seed.get('artist')}")
    print(f"\nReport-ready table written to {md_path}")


def cmd_precompute(args) -> None:
    spark = get_spark("SongRecommender-KNN-Precompute")
    model, _, count, _, fit_s = _fit(args, spark)

    started = time.perf_counter()
    recs = model.precompute_all(k=args.k, min_similarity=args.min_similarity)
    out = RESULTS_DIR / "knn_recommendations.parquet"
    recs.write.mode("overwrite").parquet(str(out))
    if args.to_mongo:
        (
            recs.write.format("mongodb")
            .mode("overwrite")
            .option("database", MONGODB_DATABASE)
            .option("collection", "knn_recommendations")
            .save()
        )
    elapsed = time.perf_counter() - started

    written = spark.read.parquet(str(out))
    covered = written.select("song_id").distinct().count()
    print(
        f"Precomputed {written.count():,} recommendations for {covered:,} of "
        f"{count:,} songs in {elapsed:.2f} s (fit {fit_s:.2f} s) -> {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Spark MLlib KNN recommender jobs.")
    parser.add_argument("--parquet", type=Path, default=PARQUET_DIR)
    parser.add_argument("--k", type=int, default=DEFAULT_RECOMMENDATION_COUNT)
    parser.add_argument("--bucket-length", type=float, default=0.5)
    parser.add_argument("--hash-tables", type=int, default=5)
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recommend", help="Recommend songs for one seed song")
    rec.add_argument("--title", required=True)
    rec.add_argument("--artist", required=True)
    rec.set_defaults(func=cmd_recommend)

    ev = sub.add_parser("evaluate", help="Timing + recall@k metrics for the report")
    ev.add_argument("--queries", type=int, default=20)
    ev.add_argument("--title")
    ev.add_argument("--artist")
    ev.set_defaults(func=cmd_evaluate)

    pre = sub.add_parser("precompute", help="Offline top-k for every song")
    pre.add_argument("--min-similarity", type=float, default=0.95)
    pre.add_argument("--to-mongo", action="store_true")
    pre.set_defaults(func=cmd_precompute)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()