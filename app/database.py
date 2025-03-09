from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import load_config

config = load_config()
DATABASE_URL = config["database_url"]


async_engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, autoflush=False, autocommit=False
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
