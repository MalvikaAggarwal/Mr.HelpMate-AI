import pickle
from sentence_transformers import CrossEncoder

CACHE_FILE = "/home/mobiledairy/Mr.HelpMate/Mr.HelpMate-AI/cache/search_cache.pkl"

import os
import pickle

def search_with_cache(query, embedder, collection, top_k=5):
    cache = {}

    # Handle missing or corrupted cache file gracefully
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("Warning: Cache file is empty or corrupted. Rebuilding cache.")
            cache = {}
    else:
        print("Cache file does not exist or is empty. Creating a new one.")

    # Return from cache if available
    if query in cache:
        return cache[query]

    # Perform embedding and retrieval
    q_embed = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[q_embed], n_results=top_k)
    docs = results["documents"][0]

    # Update and save cache
    cache[query] = docs
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    return docs


def rerank(query, docs, top_n=3):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_n]
