import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

def hybrid_search(query, documents, top_k):
    print("\n--- Running Hybrid Search ---")
    if not documents:
        return []
    
    documents = list(set(documents))  # Remove duplicates

    # BM25
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Semantic Search
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Reciprocal Rank Fusion
    k = 60
    rrf_scores = {}
    
    # Rank by BM25
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
    for rank, idx in enumerate(bm25_ranked_indices):
        if idx not in rrf_scores: rrf_scores[idx] = 0
        rrf_scores[idx] += 1 / (k + rank + 1)
        
    # Rank by Cosine
    cosine_ranked_indices = np.argsort(cosine_scores.cpu().numpy())[::-1]
    for rank, idx in enumerate(cosine_ranked_indices):
        if idx not in rrf_scores: rrf_scores[idx] = 0
        rrf_scores[idx] += 1 / (k + rank + 1)
        
    # Sort by RRF score
    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    
    top_docs = [documents[i] for i in sorted_indices[:top_k]]
    return top_docs