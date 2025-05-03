import json
from load_pdf import load_and_split
from build_index import setup_chroma
from retrieve import search_with_cache, rerank
from generate import generate_answer

# Load data
chunks = load_and_split("/workspaces/Mr.HelpMate-AI/data/Principal-Sample-Life-Insurance-Policy.pdf")
collection, embedder = setup_chroma(chunks)

import json

# Open the JSON file and load the data
with open('/home/mobiledairy/Mr.HelpMate/Mr.HelpMate-AI/queries.json', 'r') as file:
    queries = json.load(file)["queries"]


# Run RAG for each query
for query in queries:
    print("="*80)
    print(f"Query: {query}\n")

    docs = search_with_cache(query, embedder, collection)
    reranked = rerank(query, docs)

    print("Top 3 chunks:")
    for i, (text, score) in enumerate(reranked, 1):
        print(f"[{i}] {text[:300]}...\n")

    response = generate_answer(query, reranked)
    print("Generated Answer:\n", response)
