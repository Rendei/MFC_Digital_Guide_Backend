import time
from fastapi import FastAPI, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, select
from app.metrics.get_roadmap_metrics import get_average_metrics_by_model
from app.schemas import RequestModel
from app.database import get_db
from app.roadmap import generate_roadmap_batch, generate_roadmap_livetime
from app.search import hybrid_search
from app.models import SearchQuery, Roadmap, Metrics


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
    # stmt = select(Roadmap).order_by(func.random()).limit(1)
    # result = await db.execute(stmt)
    # roadmap_text = result.scalars().first().roadmap_text
    #model_name = "deepseek-ai/DeepSeek-V3-0324"

    start_time = time.time()
    
    roadmap_text, metrics = generate_roadmap_batch(data.document_id, data.user_request)
    
    generation_time = time.time() - start_time
    
    new_roadmap = Roadmap(document_id=data.document_id, user_request=data.user_request, roadmap_text=roadmap_text, model_name=data.model_name)

    new_metrics = Metrics(
        bleu_score=metrics["BLEU"],
        rouge_1_f1=metrics["ROUGE-1 F1"],
        rouge_2_f1=metrics["ROUGE-2 F1"],
        rouge_l_f1=metrics["ROUGE-L F1"],
        bert_score_f1=metrics["BERTScore F1"],
        generation_time_sec=generation_time
    )

    async with db.begin():
        db.add(new_roadmap)
        await db.flush()
        new_metrics.roadmap_id = new_roadmap.id
        db.add(new_metrics)

    return {"status": "success", "roadmap": roadmap_text}


@app.get("/model-metrics/")
async def get_model_metrics(db: AsyncSession = Depends(get_db)):
    metrics = await get_average_metrics_by_model(db)
    return {"metrics": metrics}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)