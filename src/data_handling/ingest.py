"""Stream parsed MSD records into MongoDB."""

import argparse
import time
from pathlib import Path
from typing import Iterator

from src.config import BATCH_SIZE, DATA_DIR
from src.data_handling.msd_parser import (
    iter_archive_h5,
    iter_h5_files,
    parse_h5,
)
from src.database.database import SongDatabase


def records_from_path(path: Path) -> Iterator[dict]:
    """Yield records from an HDF5 file, directory, or tar archive."""
    if path.is_file() and path.name.lower().endswith(
        (".tar.gz", ".tgz", ".tar")
    ):
        yield from iter_archive_h5(path)

    elif path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        record = parse_h5(path)
        if record:
            yield record

    elif path.is_dir():
        for h5_path in iter_h5_files(path):
            try:
                record = parse_h5(h5_path)
                if record:
                    yield record
            except (OSError, KeyError, ValueError) as exc:
                print(f"Skipping {h5_path}: {exc}")

    else:
        raise FileNotFoundError(
            f"No HDF5 directory, archive, or file found at {path}"
        )


def ingest(path: Path, batch_size: int = BATCH_SIZE) -> dict:
    """Parse songs and upsert them into MongoDB."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    started = time.perf_counter()

    with SongDatabase() as database:
        database.create_indexes()
        result = database.insert_songs(
            records_from_path(path),
            batch_size=batch_size,
        )
        validation = database.validate_ingestion()

    elapsed = time.perf_counter() - started

    print(f"Ingestion complete. Total songs: {validation['total_songs']:,}")
    print(f"Processed records: {result['processed']:,}")
    print(f"Upserted records: {result['upserted']:,}")
    print(f"Modified records: {result['modified']:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(
        "Throughput: "
        f"{result['processed'] / elapsed if elapsed else 0:.1f} records/sec"
    )
    print(f"Database validation passed: {validation['valid']}")

    return result


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest MSD HDF5 files into MongoDB."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DATA_DIR,
        help="HDF5 directory, single HDF5 file, or tar archive",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )
    args = parser.parse_args()

    ingest(args.path, args.batch_size)


if __name__ == "__main__":
    main()
