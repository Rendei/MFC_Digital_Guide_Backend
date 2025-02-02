import re
import faiss
from fastapi import FastAPI, Query, HTTPException
import numpy as np
from pydantic import BaseModel
from openai import OpenAI
import json
import time
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

document_ids = list(documents.keys())
document_texts = [" ".join(documents[doc_id]) for doc_id in document_ids]
document_names_list = [document_names.get(doc_id, "") for doc_id in document_ids]

def tokenize(text):
    return text.lower().split() 

bm25_text = BM25Okapi([tokenize(text) for text in document_texts])
bm25_names = BM25Okapi([tokenize(name) for name in document_names_list])

model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

document_embeddings = model.encode(document_texts, convert_to_numpy=True)

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

app = FastAPI()

client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

class RequestModel(BaseModel):
    document_id: str
    user_request: str


def clean_and_format_text(raw_text):
    text = raw_text.replace("\\n", "\n")

    text = re.sub(r"(?<=\n)(\d+\.|\*|-) +", r"\1 ", text)
    
    text = "\n".join(line.strip() for line in text.splitlines())

    text = re.sub(r"(?<!\n)\* ", r"\n* ", text)

    return text.strip()

@app.get("/search")
def search_documents(query: str = Query(..., description="User search query"), top_k: int = 5):
    """Hybrid search using BM25 (names + text) and embeddings"""

    bm25_text_scores = bm25_text.get_scores(tokenize(query))
    bm25_name_scores = bm25_names.get_scores(tokenize(query))

    combined_bm25_scores = 0.7 * bm25_text_scores + 0.3 * bm25_name_scores

    top_bm25_indices = np.argsort(combined_bm25_scores)[::-1][:top_k]
    top_bm25_ids = [document_ids[idx] for idx in top_bm25_indices]

    query_embedding = model.encode([query], convert_to_numpy=True)

    faiss_subset = np.array([document_embeddings[idx] for idx in top_bm25_indices])
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(faiss_subset)

    _, reranked_indices = faiss_index.search(query_embedding, top_k)

    results = [
        {
            "document_id": top_bm25_ids[idx],
            "document_name": document_names.get(top_bm25_ids[idx], "Без названия"),
            "sections": documents[top_bm25_ids[idx]]
        }
        for idx in reranked_indices[0]
    ]

    return {"query": query, "results": results}

@app.post("/generate-roadmap/")
async def generate_roadmap(data: RequestModel):
    document_id = data.document_id
    user_request = data.user_request

    document_text = documents[document_id]

    requests = [
        {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "klusterai/Meta-Llama-3.3-70B-Instruct-Turbo",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant, that provides users with step by step instructions",
                    },
                    {
                        "role": "user",
                        "content": f"Тебе предоставляется текст документа, необходимо внимательно прочитать этот текст."
                                   f"Составь дорожную карту (список задач), чтобы получить то, что хочет пользователь."
                                   f"Запрос пользователя: {user_request}, "
                                   f"Текст документа: {document_text}",
                    },
                ],
                "max_completion_tokens": 1000,
            },
        },
    ]

    file_name = "batch_input.jsonl"
    with open(file_name, "w") as file:
        for request in requests:
            file.write(json.dumps(request) + "\n")

    try:
        batch_input_file = client.files.create(file=open(file_name, "rb"), purpose="batch")

        batch_request = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        while True:
            batch_status = client.batches.retrieve(batch_request.id)

            if batch_status.status.lower() in ["completed", "failed", "cancelled"]:
                break

            time.sleep(10)  # Wait for 10 seconds before checking again

        if batch_status.status.lower() == "completed":
            result_file_id = batch_status.output_file_id
            raw_results = client.files.content(result_file_id).content

            try:
                results_json = json.loads(raw_results.decode("utf-8"))

                road_map_text = results_json["response"]["body"]["choices"][0]["message"]["content"]

                cleaned_road_map_text = clean_and_format_text(road_map_text)

                return {"status": "success", "roadmap": cleaned_road_map_text}

            except (KeyError, json.JSONDecodeError) as e:
                raise HTTPException(status_code=500, detail=f"Error processing the response: {str(e)}")

        else:
            raise HTTPException(status_code=500, detail=f"Batch failed with status: {batch_status.status}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)