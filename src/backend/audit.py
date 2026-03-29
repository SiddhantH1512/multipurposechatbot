import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

async def log_audit(
    user_id: int,
    action: str,
    resource: str,
    details: str = "",
    session: AsyncSession = None
):
    """Simple audit logging with proper datetime handling"""
    try:
        timestamp = datetime.datetime.now(datetime.timezone.utc)

        if session:
            # Use the provided session
            await session.execute(
                text("""
                    INSERT INTO audit_log (user_id, action, resource, details, timestamp)
                    VALUES (:user_id, :action, :resource, :details, :ts)
                """),
                {
                    "user_id": user_id,
                    "action": action,
                    "resource": resource,
                    "details": details[:500],
                    "ts": datetime.datetime.now(datetime.timezone.utc)
                }
            )
            await session.flush()
        else:
            # Fallback
            from src.database.engine import async_session_factory
            async with async_session_factory() as new_session:
                await new_session.execute(
                    text("""
                        INSERT INTO audit_log (user_id, action, resource, details, timestamp)
                        VALUES (:user_id, :action, :resource, :details, :ts)
                    """),
                    {
                        "user_id": user_id,
                        "action": action,
                        "resource": resource,
                        "details": details[:500],
                        "ts": timestamp
                    }
                )
                await new_session.commit()

    except Exception as e:
        print(f"[AUDIT] Failed to log {action} for user {user_id}: {e}")