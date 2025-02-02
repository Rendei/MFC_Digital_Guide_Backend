import re
import faiss
from fastapi import FastAPI, Query, HTTPException
import numpy as np
from pydantic import BaseModel
from openai import OpenAI
import json
import time
import os
from app.roadmap import generate_roadmap
from app.search import hybrid_search


app = FastAPI()

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
async def roadmap_endpoint(data: RequestModel):
    """
    Generates a roadmap (step-by-step tasks) based on a user request and a specified document.
    """
    roadmap_text = generate_roadmap(data.document_id, data.user_request)
    return {"status": "success", "roadmap": roadmap_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)