"""
build_database.py
Run this ONCE, locally, before deploying the app:

    python build_database.py

It indexes every supported audio file (.wav, .mp3, .flac, .ogg, .m4a) in
song_library/ and saves the resulting database to database.pkl. That .pkl
file is what ships with the deployed Streamlit app -- NOT the raw audio
library -- so the app works immediately without re-indexing on every restart.
"""

import pickle
from fingerprint import build_database

SONG_FOLDER = "wav_files"
OUTPUT_PATH = "database (2).pkl"

if __name__ == "__main__":
    print(f"Indexing songs in '{SONG_FOLDER}/' ...")
    db = build_database(SONG_FOLDER)
    n_songs = len(db['songs'])
    n_hashes = len(db['hashes'])
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"Indexed {n_songs} songs, {n_hashes:,} unique hashes -> saved to {OUTPUT_PATH}")
