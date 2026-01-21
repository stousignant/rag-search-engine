import json
import os
import string

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MOVIE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

def load_movies() -> list[dict]:
    with open(MOVIE_DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_DATA_PATH, "r") as f:
        return f.read().splitlines()