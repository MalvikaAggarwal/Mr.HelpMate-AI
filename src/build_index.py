import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient





def setup_chroma(chunks):
    client = PersistentClient(path="cache/chroma_db")
    collection = client.get_or_create_collection("insurance_docs")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk.page_content], ids=[str(i)])
    
    return collection, embedder
