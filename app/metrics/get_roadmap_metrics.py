from sqlalchemy import select, func
from sqlalchemy.orm import aliased
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Metrics, Roadmap


async def get_average_metrics_by_model(db: AsyncSession):
    # Create an alias for the metrics table with row numbering
    metrics_subquery = (
        select(
            Metrics,
            func.row_number().over(order_by=Metrics.id).label('row_num')
        )
        .subquery()
    )

    # Create aliases for the subquery and Roadmap
    m = aliased(Metrics, metrics_subquery)
    r = aliased(Roadmap)

    # Build the main query
    query = (
        select(
            r.model_name,
            func.avg(m.bleu_score).label('avg_bleu'),
            func.avg(m.rouge_1_f1).label('avg_rouge1'),
            func.avg(m.rouge_2_f1).label('avg_rouge2'),
            func.avg(m.rouge_l_f1).label('avg_rougeL'),
            func.avg(m.bert_score_f1).label('avg_bert_score')
        )
        .select_from(m)
        .join(r, m.roadmap_id == r.id)
        .where(metrics_subquery.c.row_num > 15)  # Start from row 15
        .group_by(r.model_name)
    )

    result = await db.execute(query)
    return result.mappings().all()