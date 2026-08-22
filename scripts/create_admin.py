"""Create an admin user. Run: python -m scripts.create_admin"""
import asyncio
from backend.app.db.session import AsyncSessionLocal, engine, Base
from backend.app.models.user import User
from backend.app.core.security import get_password_hash

async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as session:
        admin = User(
            email="admin@phishguard.local",
            hashed_password=get_password_hash("admin123"),
            is_admin=True
        )
        session.add(admin)
        await session.commit()
        print("Admin user created: admin@phishguard.local / admin123")

if __name__ == "__main__":
    asyncio.run(main())
