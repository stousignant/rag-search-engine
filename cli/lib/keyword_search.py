import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
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
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if (
            not os.path.exists(self.index_path)
            or not os.path.exists(self.docmap_path)
            or not os.path.exists(self.term_frequencies_path)
            or not os.path.exists(self.doc_lengths_path)
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
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = _parse_single_term(term)
        return self.term_frequencies.get(doc_id, Counter()).get(token, 0)

    def get_idf(self, term: str) -> float:
        token = _parse_single_term(term)
        document_count = len(self.docmap)
        document_frequency = len(self.get_documents(token))
        if document_frequency == 0:
            return 0.0
        return math.log(document_count / document_frequency)

    def get_bm25_idf(self, term: str) -> float:
        token = _parse_single_term(term)
        document_count = len(self.docmap)
        document_frequency = len(self.get_documents(token))
        return math.log((document_count - document_frequency + 0.5) / (document_frequency + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_doc_len = self.__get_avg_doc_length()
        if avg_doc_len <= 0:
            length_norm = 1.0
        else:
            length_norm = 1 - b + b * (doc_len / avg_doc_len)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] = scores.get(doc_id, 0.0) + self.bm25(doc_id, token)
        sorted_docs = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)[:limit]
        return [
            {
                "id": doc_id,
                "title": self.docmap[doc_id]["title"],
                "score": scores[doc_id],
            }
            for doc_id in sorted_docs
        ]

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)


def _load_index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.load()
    return idx


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str) -> int:
    return _load_index().get_tf(doc_id, term)


def idf_command(term: str) -> float:
    return _load_index().get_idf(term)


def bm25_idf_command(term: str) -> float:
    return _load_index().get_bm25_idf(term)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    return _load_index().get_bm25_tf(doc_id, term, k1, b)


def tfidf_command(doc_id: int, term: str) -> float:
    idx = _load_index()
    return idx.get_tf(doc_id, term) * idx.get_idf(term)


def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    return _load_index().bm25_search(query, limit)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = _load_index()

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


def _parse_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Term must tokenize to exactly one token, got {len(tokens)}: {tokens}")
    return tokens[0]


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
