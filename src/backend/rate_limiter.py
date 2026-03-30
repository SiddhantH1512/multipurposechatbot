import asyncio
from datetime import datetime, timezone
from typing import Callable

from fastapi import Depends, HTTPException, status
from redis.asyncio import Redis

from src.auth.jwt import get_current_user
from src.config import Config
from src.database.table_models import User

# Global Redis client (shared across the app)
redis_client = Redis.from_url(Config.REDIS_URL, decode_responses=True)

def _get_rate_limit_for_user(user: User) -> int:
    # (your existing logic – unchanged)
    if user.designation:
        designation_lower = user.designation.strip().lower()
        limit = Config.RATE_LIMITS_BY_DESIGNATION.get(designation_lower)
        if limit is not None:
            return limit
        for key, value in Config.RATE_LIMITS_BY_DESIGNATION.items():
            if key in designation_lower or designation_lower in key:
                return value
    if user.role:
        role_str = user.role.value if hasattr(user.role, "value") else str(user.role)
        limit = Config.RATE_LIMITS_BY_ROLE.get(role_str.upper())
        if limit is not None:
            return limit
    return Config.RATE_LIMIT_DEFAULT


def check_rate_limit(endpoint: str) -> Callable:
    async def _rate_limit_dependency(
        current_user: User = Depends(get_current_user),
    ) -> User:
        user_id = current_user.id
        limit = _get_rate_limit_for_user(current_user)
        window = Config.RATE_LIMIT_WINDOW_SECONDS
        now = datetime.now(timezone.utc).timestamp()
        key = f"rate:{user_id}:{endpoint}"

        # Redis Sorted Set sliding window
        # Remove old timestamps
        await redis_client.zremrangebyscore(key, 0, now - window)
        # Count requests in window
        count = await redis_client.zcard(key)

        if count >= limit:
            # Get oldest timestamp to calculate Retry-After
            oldest = await redis_client.zrange(key, 0, 0, withscores=True)
            retry_after = int(oldest[0][1] + window - now) + 1 if oldest else 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {limit} requests per {window}s for your tier.",
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        # Add current request
        await redis_client.zadd(key, {str(now): now})
        # Auto-expire the key after window (cleanup)
        await redis_client.expire(key, window + 10)

        return current_user

    return _rate_limit_dependency