"""
Designation-based async rate limiter using in-memory sliding window.

Uses a per-user deque of request timestamps. No external dependencies
(Redis can replace later). Thread/async-safe via asyncio.Lock.

Rate limits are tiered by user designation (job title), with fallback
to role-based limits when designation is not set.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.auth.jwt import get_current_user
from src.config import Config
from src.database.table_models import User


# ── In-memory sliding window store ──
# Key: (user_id, endpoint) → deque of UTC timestamps
_request_log: dict[tuple[int, str], deque] = defaultdict(deque)
_lock = asyncio.Lock()


def _get_rate_limit_for_user(user: User) -> int:
    """
    Determine the per-minute rate limit for a user.

    Lookup order:
        1. User's designation (job title) in RATE_LIMITS_BY_DESIGNATION
        2. User's role in RATE_LIMITS_BY_ROLE
        3. Config.RATE_LIMIT_DEFAULT

    Args:
        user: The authenticated User ORM object.

    Returns:
        Integer requests-per-minute limit.
    """
    # Try designation first (case-insensitive)
    if user.designation:
        designation_lower = user.designation.strip().lower()
        limit = Config.RATE_LIMITS_BY_DESIGNATION.get(designation_lower)
        if limit is not None:
            return limit

        # Partial match: check if any key is contained in the designation
        for key, value in Config.RATE_LIMITS_BY_DESIGNATION.items():
            if key in designation_lower or designation_lower in key:
                return value

    # Fallback to role
    if user.role:
        role_str = user.role.value if hasattr(user.role, "value") else str(user.role)
        limit = Config.RATE_LIMITS_BY_ROLE.get(role_str.upper())
        if limit is not None:
            return limit

    return Config.RATE_LIMIT_DEFAULT


def check_rate_limit(endpoint: str) -> Callable:
    """
    Factory: returns a FastAPI dependency that enforces sliding-window rate limiting.

    The returned dependency:
        1. Authenticates the user via JWT
        2. Looks up their rate limit tier by designation/role
        3. Checks requests in the last RATE_LIMIT_WINDOW_SECONDS from in-memory store
        4. Raises HTTP 429 with Retry-After header if over limit
        5. Records the request and returns the user if under limit

    Args:
        endpoint: The endpoint path string (e.g., "/chat", "/ingest") used as
                  part of the rate limit key.

    Returns:
        An async FastAPI dependency function.
    """
    async def _rate_limit_dependency(
        current_user: User = Depends(get_current_user),
    ) -> User:
        """
        Inner dependency that performs the actual rate limit check.

        Returns the User if under limit; raises HTTP 429 otherwise.
        """
        user_id = current_user.id
        limit = _get_rate_limit_for_user(current_user)
        window_seconds = Config.RATE_LIMIT_WINDOW_SECONDS
        now = datetime.now(timezone.utc)
        key = (user_id, endpoint)

        async with _lock:
            request_times = _request_log[key]

            # Prune entries outside the sliding window
            cutoff = now.timestamp() - window_seconds
            while request_times and request_times[0] < cutoff:
                request_times.popleft()

            if len(request_times) >= limit:
                # Calculate when the oldest request in the window will expire
                oldest = request_times[0]
                retry_after = int(oldest + window_seconds - now.timestamp()) + 1
                retry_after = max(retry_after, 1)

                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Max {limit} requests per {window_seconds}s for your tier.",
                    headers={"Retry-After": str(retry_after)},
                )

            # Record this request
            request_times.append(now.timestamp())

        return current_user

    return _rate_limit_dependency


def get_user_rate_info(user: User) -> dict:
    """
    Get current rate limit status for a user (for informational endpoints).

    Args:
        user: The authenticated User ORM object.

    Returns:
        Dict with limit, remaining, window_seconds, and tier info.
    """
    limit = _get_rate_limit_for_user(user)
    window_seconds = Config.RATE_LIMIT_WINDOW_SECONDS
    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - window_seconds

    # Count recent requests across all endpoints for this user
    total_recent = 0
    for (uid, _endpoint), times in _request_log.items():
        if uid == user.id:
            total_recent += sum(1 for t in times if t >= cutoff)

    return {
        "limit_per_minute": limit,
        "requests_in_window": total_recent,
        "remaining": max(0, limit - total_recent),
        "window_seconds": window_seconds,
        "designation": user.designation,
        "role": user.role.value if hasattr(user.role, "value") else str(user.role),
    }
