from sqlalchemy import Column, Integer, Float, String, JSON, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class SearchQuery(Base):
    __tablename__ = "search_queries"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    results = Column(JSON, nullable=False)
    timestamp = Column(DateTime, default=func.now())

class Roadmap(Base):
    __tablename__ = "roadmaps"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String, nullable=False)
    user_request = Column(String, nullable=False)
    roadmap_text = Column(String, nullable=False)
    model_name = Column(String, nullable=False, default="Llama-3.3-70B")
    timestamp = Column(DateTime, default=func.now())

    metrics = relationship("Metrics", back_populates="roadmap", uselist=False)

class Metrics(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    roadmap_id = Column(Integer, ForeignKey("roadmaps.id"), nullable=False)
    bleu_score = Column(Float, nullable=False)
    rouge_1_f1 = Column(Float, nullable=False)
    rouge_2_f1 = Column(Float, nullable=False)
    rouge_l_f1 = Column(Float, nullable=False)
    bert_score_f1 = Column(Float, nullable=False)
    generation_time_sec = Column(Float)
    timestamp = Column(DateTime, default=func.now())

    roadmap = relationship("Roadmap", back_populates="metrics")