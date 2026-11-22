"""MongoDB storage layer for song records."""

from collections.abc import Iterable
from typing import Any

from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection

from src.config import (
    BATCH_SIZE,
    MONGODB_COLLECTION,
    MONGODB_DATABASE,
    MONGODB_URI,
)


class SongDatabase:
    """Manage song storage and retrieval in MongoDB."""

    def __init__(
        self,
        uri: str = MONGODB_URI,
        database_name: str = MONGODB_DATABASE,
        collection_name: str = MONGODB_COLLECTION,
    ):
        self.uri = uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: MongoClient | None = None
        self.collection: Collection | None = None

    def connect(self) -> None:
        """Connect to MongoDB and check connectivity."""
        self.client = MongoClient(
            self.uri,
            serverSelectionTimeoutMS=5000,
        )
        self.client.admin.command("ping")
        self.collection = self.client[self.database_name][
            self.collection_name
        ]

    def _require_collection(self) -> Collection:
        if self.collection is None:
            raise RuntimeError("Call connect() before database operations.")
        return self.collection

    @staticmethod
    def _normalize_record(record: dict[str, Any]) -> dict | None:
        """Normalize field names and validate the record identifier."""
        document = dict(record)

        song_id = str(
            document.get("song_id") or document.get("track_id") or ""
        ).strip()

        if not song_id:
            return None

        document["song_id"] = song_id
        document["title"] = str(document.get("title") or "").strip()
        document["artist"] = str(
            document.get("artist") or document.get("artist_name") or ""
        ).strip()

        document.setdefault("artist_name", document["artist"])

        document["title_normalized"] = " ".join(
            document["title"].casefold().split()
        )
        document["artist_normalized"] = " ".join(
            document["artist"].casefold().split()
        )

        return document

    def create_indexes(self) -> None:
        """Create indexes for IDs and normalized song names."""
        collection = self._require_collection()

        collection.create_index(
            [("song_id", ASCENDING)],
            unique=True,
            name="unique_song_id",
        )
        collection.create_index(
            [("title_normalized", ASCENDING),
             ("artist_normalized", ASCENDING)],
            name="title_artist_lookup",
        )
        collection.create_index(
            [("artist_normalized", ASCENDING)],
            name="artist_lookup",
        )

    def insert_songs(
        self,
        records: Iterable[dict[str, Any]],
        batch_size: int = BATCH_SIZE,
    ) -> dict[str, int]:
        """Upsert song records in batches."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        collection = self._require_collection()
        stats = {"processed": 0, "upserted": 0, "modified": 0}
        batch = []

        def flush() -> None:
            if not batch:
                return

            operations = []

            for record in batch:
                document = self._normalize_record(record)

                if document is None:
                    continue

                operations.append(
                    UpdateOne(
                        {"song_id": document["song_id"]},
                        {"$set": document},
                        upsert=True,
                    )
                )
                stats["processed"] += 1

            if operations:
                result = collection.bulk_write(operations, ordered=False)
                stats["upserted"] += result.upserted_count
                stats["modified"] += result.modified_count

            batch.clear()

        for record in records:
            batch.append(record)

            if len(batch) >= batch_size:
                flush()

        flush()
        return stats

    def find_song(self, song_id: str) -> dict | None:
        """Find a song by its canonical song ID."""
        return self._require_collection().find_one(
            {"song_id": str(song_id)},
            {"_id": 0},
        )

    def find_by_title_artist(
        self,
        title: str,
        artist: str,
    ) -> dict | None:
        """Look up a song using normalized title and artist names."""
        normalize = lambda value: " ".join(value.casefold().split())

        return self._require_collection().find_one({
            "title_normalized": normalize(title),
            "artist_normalized": normalize(artist),
        })

    def get_song_features(
        self,
        fields: list[str] | None = None,
    ) -> list[dict]:
        """Retrieve song records for feature preparation."""
        fields = fields or [
            "song_id", "track_id", "title", "artist",
            "duration", "end_of_fade_in", "loudness", "tempo",
            "key", "key_confidence", "mode", "mode_confidence",
            "time_signature", "time_signature_confidence",
            "artist_familiarity", "artist_hotness", "song_hotness", "year",
        ]

        projection = {field: 1 for field in fields}
        projection["_id"] = 0

        return list(self._require_collection().find({}, projection))

    def validate_ingestion(self) -> dict:
        """Return collection counts and basic data quality checks."""
        collection = self._require_collection()
        total = collection.count_documents({})
        missing_id = collection.count_documents({
            "$or": [
                {"song_id": {"$exists": False}},
                {"song_id": None},
                {"song_id": ""},
            ]
        })
        titled_and_attributed = collection.count_documents({
            "title": {"$type": "string", "$ne": ""},
            "artist": {"$type": "string", "$ne": ""},
        })

        return {
            "total_songs": total,
            "missing_song_id": missing_id,
            "songs_with_title_and_artist": titled_and_attributed,
            "valid": total > 0 and missing_id == 0,
        }

    def close(self) -> None:
        """Close the client."""
        if self.client is not None:
            self.client.close()

        self.client = None
        self.collection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
