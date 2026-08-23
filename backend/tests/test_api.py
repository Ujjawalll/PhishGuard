import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app, lifespan

@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_auth_and_scan():
    async with lifespan(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Register
            import uuid
            email = f"test_{uuid.uuid4()}@example.com"
            res = await ac.post("/auth/register", json={"email": email, "password": "password"})
            assert res.status_code == 200
            token = res.json()["access_token"]
            
            # Scan (Fast)
            headers = {"Authorization": f"Bearer {token}"}
            scan_res = await ac.post("/scan/", json={"url": "http://example.com"}, headers=headers)
            assert scan_res.status_code == 200
            data = scan_res.json()
            assert "risk_level" in data
            assert "scan_id" in data
            
            # Scan (Deep)
            deep_res = await ac.post("/scan/deep", json={"url": "http://example.com"}, headers=headers)
            assert deep_res.status_code == 200
            deep_data = deep_res.json()
            assert "risk_level" in deep_data
