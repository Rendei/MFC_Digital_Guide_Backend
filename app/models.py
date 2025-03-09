from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, func
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
    timestamp = Column(DateTime, default=func.now())
