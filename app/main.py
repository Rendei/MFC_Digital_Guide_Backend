from fastapi import FastAPI, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas import RequestModel
from app.database import get_db
from app.roadmap import generate_roadmap_batch, generate_roadmap_livetime
from app.search import hybrid_search
from app.models import SearchQuery, Roadmap


app = FastAPI()

@app.get("/search")
async def search_documents(query: str = Query(..., description="User search query"), top_k: int = 5, db: AsyncSession = Depends(get_db)):
    """
    Hybrid search endpoint that uses BM25 and FAISS for re-ranking.
    """
    results = hybrid_search(query, top_k)
    new_search = SearchQuery(query=query, results=results)
    
    db.add(new_search)
    await db.commit()

    return {"query": query, "results": results}


@app.post("/generate-roadmap/")
async def roadmap_endpoint(data: RequestModel, db: AsyncSession = Depends(get_db)):
    """
    Generates a roadmap (step-by-step tasks) based on a user request and a specified document.
    """
    roadmap_text = generate_roadmap_batch(data.document_id, data.user_request)
    new_roadmap = Roadmap(document_id=data.document_id, user_request=data.user_request, roadmap_text=roadmap_text)
    
    db.add(new_roadmap)
    await db.commit()
    
    return {"status": "success", "roadmap": roadmap_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)