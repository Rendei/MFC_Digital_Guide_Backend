import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from app.data_loader import load_documents, load_document_names
from app.utils import tokenize


documents = load_documents()
document_names = load_document_names()

document_ids = list(documents.keys())
document_texts = [" ".join(documents[doc_id]) for doc_id in document_ids]
document_names_list = [document_names.get(doc_id, "") for doc_id in document_ids]

bm25_text = BM25Okapi([tokenize(text) for text in document_texts])
bm25_names = BM25Okapi([tokenize(name) for name in document_names_list])

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
document_embeddings = model.encode(document_texts, convert_to_numpy=True)

dimension = document_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(document_embeddings)


def hybrid_search(query: str, top_k: int = 5):
    bm25_text_scores = bm25_text.get_scores(tokenize(query))
    bm25_name_scores = bm25_names.get_scores(tokenize(query))
    combined_bm25_scores = 0.7 * bm25_text_scores + 0.3 * bm25_name_scores

    top_bm25_indices = np.argsort(combined_bm25_scores)[::-1][:top_k]
    top_bm25_ids = [document_ids[idx] for idx in top_bm25_indices]

    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss_subset = np.array([document_embeddings[idx] for idx in top_bm25_indices])

    temp_index = faiss.IndexFlatL2(dimension)
    temp_index.add(faiss_subset)
    _, reranked_indices = temp_index.search(query_embedding, top_k)

    results = [
        {
            "document_id": top_bm25_ids[idx],
            "document_name": document_names.get(top_bm25_ids[idx], "Без названия"),
            "sections": documents[top_bm25_ids[idx]]
        }
        for idx in reranked_indices[0]
    ]
    return results
