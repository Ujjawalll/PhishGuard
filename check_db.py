import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

DATABASE_URL = "postgresql+asyncpg://phishguard:phishguard_secret@localhost:5432/phishguard_db"

async def main():
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    
    async with async_session() as session:
        result = await session.execute(text("SELECT url, risk_level, ml_probability, rule_score, explanation_json FROM scans ORDER BY timestamp DESC LIMIT 5"))
        for row in result:
            print(f"URL: {row[0]}")
            print(f"Risk: {row[1]} | ML: {row[2]} | Rule: {row[3]}")
            print(f"Explanation: {row[4][:100]}...\n")

asyncio.run(main())
