# Song-Recommender-AI

A Python song recommender using MongoDB and the Million Song Dataset (MSD). It recursively parses the official dataset's nested HDF5 layout and ranks songs using standardized audio-analysis features and various clustering/similarity algorithms.

## Info
**Team:**
- Itay Shemesh, DBSCAN 
- Felix Roman, KNN
- Connor Chase, GMM
- Jaden Bryant, Hierarchical Clustering

## Dataset
- **Compressed:** https://www.kaggle.com/datasets/ryanholbrook/the-million-songs-dataset
- **Full:** http://millionsongdataset.com/

## Prerequisites
- **Python 3.12**
- **MongoDB** (running locally on port 27017, or accessible via network)

## Setup & Installation

1. **Clone the repository and enter the directory:**
   ```bash
   git clone <repository_url>
   cd Song-Recomender-AI
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   python3.12 -m pip install -r requirements.txt
   ```

4. **Start MongoDB:**
   Ensure you have a MongoDB instance running. You can run one quickly locally using Docker:
   ```bash
   docker run -d -p 27017:27017 --name mongodb mongo
   ```

5. **Download the Dataset:**
   Place your downloaded Kaggle dataset (`.h5` files, folders, or `.tar.gz` archives) inside the `data/` folder.

## Run with Docker Compose

Start MongoDB and build the app:

```bash
docker compose up -d mongo
docker compose build app
```

Compose publishes MongoDB on `localhost:27018` by default, avoiding a conflict
with a MongoDB service already using port `27017`. Set `MONGODB_HOST_PORT` to
change the host port if needed. The app container connects to MongoDB over the
Compose network on port `27017`.

Place the Million Song Dataset HDF5 files in `data/`. For a tar archive, you may also put the whole archive file there. (e.g., `data/msd.tar.gz`.)

```bash
docker compose run --rm app python -m src.data_handling.ingest --path data/
docker compose run --rm app python -m src.database.validate_db
docker compose run --rm app python -m src.main --title "Song Title" --artist "Artist Name"
```

## Configuration
The application is configured using environment variables, with defaults managed in `src/config.py`:
- `MONGODB_URI`: Connection string for MongoDB (default: `mongodb://localhost:27017/`)
- `MONGODB_DATABASE`: Database name (default: `song_recommender`)
- `MONGODB_COLLECTION`: Collection name for songs (default: `songs`)

## Usage Pipeline

### 1. Data Ingestion
Parse the HDF5 dataset and upsert the records into MongoDB:
```bash
python -m src.data_handling.ingest --path data/
```

### 2. Validation
Verify the MongoDB contents, checking connectivity, document counts, required metadata, and indexes:
```bash
python -m src.database.validate_db
```

### 3. Running the Recommender
The DBSCAN CLI reads the songs collection from MongoDB through the Spark connector and returns up to 10 recommendations for the given song title and artist. Parquet export is not required for DBSCAN:
```bash
python -m src.main --title "Song Title" --artist "Artist Name"
```

Tune DBSCAN with `--eps`, `--min-samples`, and `--recommendations` as needed.

*(Alternatively, you can run `./start.sh` to see a quick overview of the available project commands).*

## Recommender Models
We implement four different machine learning approaches to compare recommendation quality:
- **KNN (K-Nearest Neighbors):** Finds the closest songs in the feature space using cosine similarity.
- **DBSCAN:** Identifies density-connected clusters of songs.
- **GMM (Gaussian Mixture Model):** Assigns songs to probabilistic Gaussian components.
- **Hierarchical Clustering:** Uses bisecting k-means to split songs into clusters, then ranks tracks in the seed's cluster by Euclidean distance on standardized features.

## Data Preparation & Schema
- **Reduction:** Selects identity fields, year/release, and a compact set of useful numeric analysis features from the HDF5 dataset.
- **Cleansing:** Skips unreadable HDF5 files and records without title or artist; decodes byte strings; normalizes whitespace/case for lookup; converts non-finite numbers to null.
- **Transformation:** Extracts scalar values from MSD HDF5 groups; standardizes numeric fields handling missing values via median imputation.
- **Storage:** Uses one MongoDB document per song and bulk upserts to support repeatable ingestion.

### MongoDB Document Schema
Each song document in MongoDB is structured with the following key fields:
- `song_id`: string (Unique identifier for the song)
- `track_id`: string (Unique identifier for the specific track version)
- `title`: string (Song title)
- `artist`: string (Artist name)
- `title_normalized`: string (Lowercase, whitespace-stripped title for searches)
- `artist_normalized`: string (Lowercase, whitespace-stripped artist for searches)
- `release`: string (Album release name)
- `artist_id`: string (Unique identifier for the artist)
- `duration`, `tempo`, `loudness`: float (Audio features)
- `key`, `mode`, `time_signature`: float (Musical features)
- `key_confidence`, `mode_confidence`, `time_signature_confidence`: float
- `artist_familiarity`, `artist_hotness`, `song_hotness`: float (Popularity metrics)
- `year`: float (Release year)

## Credit
```text
Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. 
The Million Song Dataset. In Proceedings of the 12th International Society
for Music Information Retrieval Conference (ISMIR 2011), 2011.
```
