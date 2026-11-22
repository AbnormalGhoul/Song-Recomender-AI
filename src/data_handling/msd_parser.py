"""Parse song metadata and analysis features from MSD HDF5 files."""

import io
import tarfile
from pathlib import Path
from typing import Iterator

import h5py

from src.config import DATA_DIR


def _decode(value):
    """Convert HDF5 values into ordinary Python values."""
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()

    return value


def _read_song(h5: h5py.File) -> dict:
    """Extract available metadata and analysis fields from an open HDF5 file."""
    record = {}

    metadata = h5.get("metadata/songs")
    analysis = h5.get("analysis/songs")

    if metadata is not None and len(metadata) > 0:
        fields = metadata.dtype.names or ()

        for field in (
            "track_id",
            "song_id",
            "title",
            "artist_name",
            "release",
            "artist_id",
            "artist_familiarity",
            "artist_hotness",
            "song_hotness",
        ):
            if field in fields:
                record[field] = _decode(metadata[field][0])

    if analysis is not None and len(analysis) > 0:
        fields = analysis.dtype.names or ()

        for field in (
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
            "year",
        ):
            if field in fields:
                value = _decode(analysis[field][0])

                try:
                    record[field] = float(value)
                except (TypeError, ValueError):
                    record[field] = None

    # Keep consistent names for database queries and recommendation code.
    record["title"] = str(record.get("title") or "").strip()
    record["artist"] = str(
        record.get("artist_name") or ""
    ).strip()

    record["title_normalized"] = " ".join(
        record["title"].casefold().split()
    )
    record["artist_normalized"] = " ".join(
        record["artist"].casefold().split()
    )

    # The song ID is the preferred database identifier.
    record["song_id"] = str(record.get("song_id") or "").strip()
    record["track_id"] = str(record.get("track_id") or "").strip()

    return record


def parse_h5(file_path: Path | str) -> dict | None:
    """Parse one HDF5 file into a MongoDB-compatible dictionary."""
    with h5py.File(file_path, "r") as h5:
        record = _read_song(h5)

    if not record.get("song_id") and not record.get("track_id"):
        return None

    return record


def iter_h5_files(directory: Path | str) -> Iterator[Path]:
    """Yield HDF5 files recursively from a directory."""
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
            yield path


def iter_archive_h5(archive_path: Path | str) -> Iterator[dict]:
    """Parse HDF5 files inside a tar archive without extracting them."""
    archive_path = Path(archive_path)

    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue

            if Path(member.name).suffix.lower() not in {".h5", ".hdf5"}:
                continue

            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue

            try:
                content = file_obj.read()

                with h5py.File(io.BytesIO(content), "r") as h5:
                    record = _read_song(h5)

                if record.get("song_id") or record.get("track_id"):
                    yield record

            except (OSError, KeyError, ValueError) as exc:
                print(f"Skipping archive member {member.name}: {exc}")


class MillionSongParser:
    """Convenience interface for working with a dataset directory."""

    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)

    def find_h5_files(self) -> list[Path]:
        """Return all HDF5 files in the dataset directory."""
        if not self.data_dir.exists():
            return []
        return list(iter_h5_files(self.data_dir))

    def parse_h5_file(self, file_path: Path | str) -> dict | None:
        """Parse a single HDF5 file."""
        return parse_h5(file_path)

    def iter_songs(self) -> Iterator[dict]:
        """Yield parsed records one at a time."""
        for path in self.find_h5_files():
            try:
                record = parse_h5(path)
                if record:
                    yield record
            except (OSError, KeyError, ValueError) as exc:
                print(f"Skipping {path}: {exc}")

    def load_dataset(self, limit: int | None = None):
        """Load parsed records into a pandas DataFrame."""
        import pandas as pd

        records = []

        for record in self.iter_songs():
            records.append(record)

            if limit is not None and len(records) >= limit:
                break

        return pd.DataFrame.from_records(records)
