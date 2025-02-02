import re
import faiss
from fastapi import FastAPI, Query, HTTPException
import numpy as np
from pydantic import BaseModel
from openai import OpenAI
import json
import time
import os
from app.search import hybrid_search


app = FastAPI()

client = OpenAI(
    base_url="https://api.kluster.ai/v1",
    api_key=api_key,
)

class RequestModel(BaseModel):
    document_id: str
    user_request: str


@app.get("/search")
def search_documents(query: str = Query(..., description="User search query"), top_k: int = 5):
    """
    Hybrid search endpoint that uses BM25 and FAISS for re-ranking.
    """
    results = hybrid_search(query, top_k)
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