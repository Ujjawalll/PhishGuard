import json
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc

from backend.app.db.session import get_db
from backend.app.models.scan import Scan
from backend.app.models.user import User
from backend.app.api.deps import get_current_admin_user

router = APIRouter()

# ── WebSocket connection manager ──────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                pass  # Connection may have closed

manager = ConnectionManager()

@router.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket):
    """
    Real-time scan event feed.
    Admin authenticates by sending JWT as first message after connection.
    """
    await manager.connect(websocket)
    try:
        # Wait for auth token as first message
        token = await websocket.receive_text()
        # Validate token (reuse existing JWT logic)
        from jose import jwt, JWTError
        from backend.app.core.config import settings
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id = payload.get("sub")
            if not user_id:
                await websocket.close(code=4001, reason="Invalid token")
                return
        except JWTError:
            await websocket.close(code=4001, reason="Invalid token")
            return

        # Keep connection alive
        while True:
            await websocket.receive_text()  # Client pings
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── REST endpoints ────────────────────────────────────────────────────

@router.get("/stats")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """Aggregate scan statistics for the dashboard."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)

    # Total scans
    total_result = await db.execute(select(func.count(Scan.id)))
    total_scans = total_result.scalar()

    # Today's scans
    today_result = await db.execute(
        select(func.count(Scan.id)).where(Scan.timestamp >= today_start)
    )
    today_scans = today_result.scalar()

    # This week's scans
    week_result = await db.execute(
        select(func.count(Scan.id)).where(Scan.timestamp >= week_start)
    )
    week_scans = week_result.scalar()

    # Risk distribution (all time)
    risk_result = await db.execute(
        select(Scan.risk_level, func.count(Scan.id)).group_by(Scan.risk_level)
    )
    risk_distribution = {row[0]: row[1] for row in risk_result.all()}

    # Stage distribution
    stage_result = await db.execute(
        select(Scan.stage, func.count(Scan.id)).group_by(Scan.stage)
    )
    stage_distribution = {row[0]: row[1] for row in stage_result.all()}

    # Average scores
    try:
        avg_result = await db.execute(
            select(
                func.avg(Scan.ml_probability),
                func.avg(Scan.rule_score),
                func.avg(Scan.fused_score)
            )
        )
        avgs = avg_result.one()
    except Exception:
        avgs = (0, 0, 0)

    # Active users count
    user_result = await db.execute(select(func.count(User.id)).where(User.is_active == True))
    active_users = user_result.scalar()

    return {
        "total_scans": total_scans,
        "today_scans": today_scans,
        "week_scans": week_scans,
        "risk_distribution": risk_distribution,
        "stage_distribution": stage_distribution,
        "avg_ml_probability": round(float(avgs[0] or 0), 4),
        "avg_rule_score": round(float(avgs[1] or 0), 4),
        "avg_fused_score": round(float(avgs[2] or 0), 4),
        "active_users": active_users
    }

@router.get("/alerts")
async def get_recent_alerts(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user),
    limit: int = 50
):
    """Recent HIGH_RISK and SUSPICIOUS scans."""
    result = await db.execute(
        select(Scan)
        .where(Scan.risk_level.in_(["HIGH_RISK", "SUSPICIOUS"]))
        .order_by(desc(Scan.timestamp))
        .limit(limit)
    )
    scans = result.scalars().all()

    return [
        {
            "scan_id": s.id,
            "url": s.url,
            "domain": s.domain,
            "risk_level": s.risk_level,
            "fused_score": s.fused_score,
            "stage": s.stage,
            "triggered_rules": json.loads(s.triggered_rules_json) if s.triggered_rules_json else [],
            "timestamp": s.timestamp.isoformat() if s.timestamp else None
        }
        for s in scans
    ]

@router.get("/top-rules")
async def get_top_triggered_rules(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user),
    limit: int = 100
):
    """
    Aggregate which rules are triggered most frequently.
    Reads triggered_rules_json from recent scans.
    """
    result = await db.execute(
        select(Scan.triggered_rules_json)
        .where(Scan.triggered_rules_json.isnot(None))
        .order_by(desc(Scan.timestamp))
        .limit(limit)
    )
    rows = result.scalars().all()

    rule_counts: dict[str, int] = {}
    for row in rows:
        try:
            rules = json.loads(row)
            for rule in rules:
                rid = rule.get("rule_id", "unknown")
                rule_counts[rid] = rule_counts.get(rid, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Sort by count descending
    sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"rule_id": k, "count": v} for k, v in sorted_rules]

@router.get("/health")
async def admin_health(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(get_current_admin_user)
):
    """System health check for the admin dashboard."""
    import backend.app.api.scan as scan_module

    return {
        "api": "ok",
        "database": "ok",  # If we got here, DB is working
        "ml_model_loaded": scan_module.ml_pipeline is not None,
        "rule_engine_loaded": scan_module.rule_engine is not None,
        "explainer_loaded": scan_module.explainer is not None
    }
