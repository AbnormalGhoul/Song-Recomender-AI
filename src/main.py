import argparse
import sys
import pandas as pd
from src.database.database import SongDatabase
from src.recommenders.knn import KNNRecommender
from src.recommenders.dbscan import DBSCANRecommender
from src.recommenders.gmm import GMMRecommender
from src.recommenders.hierarchical import HierarchicalRecommender

def main():
    parser = argparse.ArgumentParser(description="Song Recommender AI")
    parser.add_argument("--title", required=True, help="Song title")
    parser.add_argument("--artist", required=True, help="Song artist")
    args = parser.parse_args()

    try:
        with SongDatabase() as db:
            song = db.find_by_title_artist(args.title, args.artist)
            if not song:
                print(f"Error: Song '{args.title}' by '{args.artist}' not found in the database.")
                sys.exit(1)
            
            track_id = song.get("track_id") or song.get("song_id")
            
            print(f"Found song: {song.get('title', args.title)} by {song.get('artist', args.artist)} (ID: {track_id})")
            print("Loading dataset features from database...")
            records = db.get_song_features()
            if not records:
                print("Error: No songs found in the database. Please ingest data first.")
                sys.exit(1)
            
            songs_df = pd.DataFrame.from_records(records)
    except Exception as e:
        print(f"Database error: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(songs_df)} songs. Training models...\n")

    recommenders = {
        "KNN": KNNRecommender(),
        "DBSCAN": DBSCANRecommender(),
        "GMM": GMMRecommender(n_components=min(10, len(songs_df))),
        "Hierarchical": HierarchicalRecommender(n_clusters=min(10, len(songs_df))),
    }

    for name, recommender in recommenders.items():
        print(f"--- {name} Recommendations ---")
        try:
            recommender.fit(songs_df)
            recs = recommender.recommend(track_id=track_id)
            if recs.empty:
                print("No recommendations found.")
            else:
                for i, (_, row) in enumerate(recs.iterrows(), 1):
                    title = row.get("title", "Unknown")
                    artist = row.get("artist", "Unknown")
                    print(f"{i}. {title} by {artist}")
        except Exception as e:
            print(f"Failed to generate recommendations: {e}")
        print()

if __name__ == "__main__":
    main()
