import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import CACHE_DIR, load_movies


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text must not be empty or contain only whitespace.")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(os.path.join(CACHE_DIR, "movie_embeddings.npy"), self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        cache_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
        if os.path.exists(cache_path):
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = 5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `verify_embeddings` command first.")
        
        query_embedding = self.generate_embedding(query)
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, doc in similarities[:limit]:
            results.append({
                "score": similarity,
                "title": doc["title"],
                "description": doc["description"]
            })
        
        return results

def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")