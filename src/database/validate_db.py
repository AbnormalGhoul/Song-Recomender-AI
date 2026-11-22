"""Print a simple MongoDB ingestion validation report."""

from src.database.database import SongDatabase


def main() -> None:
    """Connect to MongoDB and report ingestion status."""
    with SongDatabase() as database:
        collection = database.collection
        assert collection is not None

        print("MongoDB ping: OK")

        report = database.validate_ingestion()
        print(f"Song count: {report['total_songs']:,}")
        print(f"Missing song IDs: {report['missing_song_id']:,}")
        print(
            "Songs with title and artist: "
            f"{report['songs_with_title_and_artist']:,}"
        )

        print("\nSample records:")
        projection = {
            "_id": 0,
            "song_id": 1,
            "title": 1,
            "artist": 1,
            "year": 1,
        }

        for song in collection.find({}, projection).limit(5):
            print(song)

        print("\nIndexes:")
        print(list(collection.index_information().keys()))

        print(f"\nValidation passed: {report['valid']}")


if __name__ == "__main__":
    main()
