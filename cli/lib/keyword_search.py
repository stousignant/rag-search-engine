import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        if (
            not os.path.exists(self.index_path)
            or not os.path.exists(self.docmap_path)
            or not os.path.exists(self.term_frequencies_path)
        ):
            raise FileNotFoundError(
                "Index not found. Run the build command first to create the index."
            )
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Term must tokenize to exactly one token, got {len(tokens)}: {tokens}")
        token = tokens[0]
        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Term must tokenize to exactly one token, got {len(tokens)}: {tokens}")
    token = tokens[0]
    num_documents = len(idx.docmap)
    document_frequency = len(idx.get_documents(token))
    if document_frequency == 0:
        return 0.0
    return math.log(num_documents / document_frequency)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    tf = idx.get_tf(doc_id, term)
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Term must tokenize to exactly one token, got {len(tokens)}: {tokens}")
    token = tokens[0]
    num_documents = len(idx.docmap)
    document_frequency = len(idx.get_documents(token))
    idf = math.log(num_documents / document_frequency) if document_frequency > 0 else 0.0
    return tf * idf


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()

    query_tokens = tokenize_text(query)
    results = []
    seen: set[int] = set()

    for token in query_tokens:
        for doc_id in idx.get_documents(token):
            if doc_id not in seen:
                seen.add(doc_id)
                results.append(idx.docmap[doc_id])
                if len(results) >= limit:
                    return results

    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split(" ")
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    valid_tokens = filter_stop_words(valid_tokens)
    valid_tokens = stem_words(valid_tokens)
    return valid_tokens

def filter_stop_words(words: list[str]) -> list[str]:
    stop_words = load_stop_words()
    filtered_words = []
    for word in words:
        if word not in stop_words:
            filtered_words.append(word)
    return filtered_words
                
def stem_words(words: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words
