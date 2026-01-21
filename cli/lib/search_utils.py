import json
import os
import string

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3

BM25_K1 = 1.5   # Tunable parameter that controls the diminishing returns of term frequency
BM25_B = 0.75   # Tunable parameter that controls how much we care about document length

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0

SEMANTIC_CHUNK_MAX_SIZE = 4
SEMANTIC_CHUNK_OVERLAP = 0


def load_movies() -> list[dict]:
    with open(MOVIE_DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_DATA_PATH, "r") as f:
        return f.read().splitlines()