from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import load_config

config = load_config()
DATABASE_URL = config["database_url"]


engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    with SessionLocal() as session:
        yield session
