"""Benchmark DBSCAN fit and repeated recommendation queries."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from functools import reduce
from operator import add
from pathlib import Path

import pyspark
from pyspark.sql import functions as F

from src.config import RESULTS_DIR, SPARK_SHUFFLE_PARTITIONS
from src.recommenders.dbscan import DBSCANRecommender
from src.spark.session import get_spark


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
    }


def _identity_column(columns: list[str]) -> str:
    return "song_id" if "song_id" in columns else "track_id"


def _nonempty_identifier(column: str):
    value = F.trim(F.col(column).cast("string"))
    return F.when(value.isNotNull() & (F.length(value) > 0), value)


def _container_limits() -> dict[str, float | int | None]:
    """Read cgroup v2 CPU and memory limits when the runtime exposes them."""
    limits: dict[str, float | int | None] = {
        "cpu_quota_cores": None,
        "memory_limit_bytes": None,
    }
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            limits["cpu_quota_cores"] = int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        memory = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if memory != "max":
            limits["memory_limit_bytes"] = int(memory)
    except (OSError, ValueError):
        pass
    return limits


def benchmark(
    query_count: int = 20,
    recommendation_count: int = 10,
    eps: float = 0.5,
    min_samples: int = 5,
    sample_seed: int = 42,
) -> dict:
    """Fit DBSCAN once, then compare repeated queries to an exact full scan."""
    if query_count < 1:
        raise ValueError("query_count must be positive.")

    spark = get_spark("SongRecommender-DBSCAN-Benchmark")
    recommender = DBSCANRecommender(
        eps=eps,
        min_samples=min_samples,
        recommendation_count=recommendation_count,
        spark=spark,
    )

    try:
        fit_started = time.perf_counter()
        recommender.fit_from_mongodb()
        fit_seconds = time.perf_counter() - fit_started
        songs = recommender.songs
        if songs is None:
            raise RuntimeError("DBSCAN fit completed without a scored song table.")

        total_songs = songs.count()
        id_column = _identity_column(songs.columns)
        if id_column not in songs.columns:
            raise ValueError("MongoDB songs need a song_id or track_id column.")

        seed_expression = _nonempty_identifier("track_id") if "track_id" in songs.columns else None
        if seed_expression is None:
            seed_expression = _nonempty_identifier("song_id")
        if "song_id" in songs.columns and "track_id" in songs.columns:
            seed_expression = F.coalesce(
                _nonempty_identifier("track_id"),
                _nonempty_identifier("song_id"),
            )

        seed_rows = (
            songs.withColumn("__benchmark_seed_id", seed_expression)
            .filter(F.col("__benchmark_seed_id").isNotNull())
            .select("__benchmark_seed_id")
            .orderBy(F.xxhash64("__benchmark_seed_id", F.lit(sample_seed)))
            .limit(query_count)
            .collect()
        )
        if not seed_rows:
            raise ValueError("No songs with a usable identifier were found.")

        seeds = [row["__benchmark_seed_id"] for row in seed_rows]
        feature_to_scaled = dict(zip(
            recommender.feature_columns,
            recommender._scaled_columns,
        ))

        def exact_recommendations(seed_id: str) -> list[str]:
            seed = recommender.find_song(seed_id)
            squared_distances = [
                (F.col(scaled) - F.lit(float(seed[scaled]))) ** 2
                for scaled in feature_to_scaled.values()
            ]
            distance = reduce(add, squared_distances)
            candidates = songs
            for field in ("song_id", "track_id"):
                if field in songs.columns:
                    candidates = candidates.filter(
                        F.col(field).cast("string") != seed_id
                    )
            rows = (
                candidates.withColumn("__exact_distance", distance)
                .orderBy(F.col("__exact_distance"), F.col(id_column))
                .select(id_column)
                .limit(recommendation_count)
                .collect()
            )
            return [str(row[id_column]) for row in rows]

        # Warm the cached song table and Spark SQL path before timed queries.
        warmup_seed = seeds[0]
        recommender.recommend(warmup_seed, n=recommendation_count)
        exact_recommendations(warmup_seed)

        dbscan_latencies = []
        exact_latencies = []
        recalls = []
        candidate_counts = []
        seed_results = []

        for seed_id in seeds:
            target = recommender.find_song(seed_id)
            cluster = int(target["cluster"])
            if cluster == -1:
                candidates = 0
            else:
                candidates = (
                    songs.filter(F.col("cluster") == cluster).count() - 1
                )
            candidate_counts.append(max(0, candidates))

            started = time.perf_counter()
            recommended = recommender.recommend(seed_id, n=recommendation_count)
            dbscan_seconds = time.perf_counter() - started

            started = time.perf_counter()
            exact_ids = exact_recommendations(seed_id)
            exact_seconds = time.perf_counter() - started

            result_column = _identity_column(list(recommended.columns))
            recommended_ids = (
                recommended[result_column].astype(str).tolist()
                if result_column in recommended.columns
                else []
            )
            overlap = len(set(recommended_ids) & set(exact_ids))
            recall = overlap / len(exact_ids) if exact_ids else 0.0

            dbscan_latencies.append(dbscan_seconds)
            exact_latencies.append(exact_seconds)
            recalls.append(recall)
            seed_results.append({
                "seed_id": seed_id,
                "cluster": cluster,
                "recommendations_returned": len(recommended_ids),
                "dbscan_query_seconds": dbscan_seconds,
                "exact_query_seconds": exact_seconds,
                "recall_at_n": recall,
            })

        java_version = spark._jvm.java.lang.System.getProperty("java.version")
        non_noise_results = [
            query for query in seed_results if query["cluster"] != -1
        ]
        non_noise_recalls = [query["recall_at_n"] for query in non_noise_results]
        runtime = {
            "python": platform.python_version(),
            "pyspark": pyspark.__version__,
            "spark": spark.version,
            "java": java_version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "spark_master": spark.sparkContext.master,
            "spark_default_parallelism": spark.sparkContext.defaultParallelism,
            "spark_shuffle_partitions": SPARK_SHUFFLE_PARTITIONS,
        }
        runtime.update(_container_limits())
        return {
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "Spark RDD DBSCAN",
            "dataset": {
                "source": "MongoDB via MongoDB Spark Connector",
                "songs": total_songs,
                "features": recommender.feature_columns,
            },
            "parameters": {
                "eps": eps,
                "min_samples": min_samples,
                "recommendations": recommendation_count,
                "queries": len(seeds),
                "sample_seed": sample_seed,
            },
            "runtime": runtime,
            "fit_seconds": fit_seconds,
            "cluster_count": songs.filter(F.col("cluster") >= 0)
            .select("cluster").distinct().count(),
            "noise_songs": songs.filter(F.col("cluster") == -1).count(),
            "dbscan_query_latency_seconds": _summary(dbscan_latencies),
            "exact_full_scan_query_latency_seconds": _summary(exact_latencies),
            "mean_recall_at_n_against_exact": statistics.fmean(recalls),
            "min_recall_at_n_against_exact": min(recalls),
            "non_noise_queries": len(non_noise_results),
            "mean_recall_at_n_non_noise": (
                statistics.fmean(non_noise_recalls) if non_noise_recalls else None
            ),
            "min_recall_at_n_non_noise": (
                min(non_noise_recalls) if non_noise_recalls else None
            ),
            "noise_queries": len(seed_results) - len(non_noise_results),
            "mean_cluster_candidates_per_query": statistics.fmean(candidate_counts),
            "mean_exact_candidates_per_query": max(0, total_songs - 1),
            "mean_candidate_reduction_fraction": (
                1.0 - statistics.fmean(candidate_counts) / max(1, total_songs - 1)
            ),
            "non_noise_dbscan_query_latency_seconds": (
                _summary([
                    query["dbscan_query_seconds"] for query in non_noise_results
                ])
                if non_noise_results else None
            ),
            "non_noise_exact_query_latency_seconds": (
                _summary([
                    query["exact_query_seconds"] for query in non_noise_results
                ])
                if non_noise_results else None
            ),
            "warmup_queries_excluded": 1,
            "measurement_notes": {
                "fit_seconds": "MongoDB read, preprocessing, clustering, and result caching.",
                "query_latency": "Warm-cache seed lookup through recommendation collection.",
                "exact_latency": "Warm-cache seed lookup, full-catalog distance sort, and collection.",
                "recall_at_n": "Overlap with exact full-catalog top-N divided by the exact result count.",
                "noise_queries": "Noise seeds return no DBSCAN recommendations and score zero recall.",
            },
            "queries": seed_results,
        }
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Spark DBSCAN recommendations against exact scan."
    )
    parser.add_argument("--queries", type=int, default=20)
    parser.add_argument("--recommendations", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "dbscan_benchmark.json",
    )
    args = parser.parse_args()

    result = benchmark(
        query_count=args.queries,
        recommendation_count=args.recommendations,
        eps=args.eps,
        min_samples=args.min_samples,
        sample_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Benchmark results saved to {args.output}")


if __name__ == "__main__":
    main()
