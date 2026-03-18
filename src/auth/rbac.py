"""
Role-Based Access Control (RBAC) dependency helpers for FastAPI.

Provides factory functions and convenience shorthands for enforcing
role-based access on API endpoints via FastAPI's dependency injection.
"""

from fastapi import Depends, HTTPException, status
from src.auth.jwt import get_current_user
from src.database.table_models import User, UserRole
from typing import List


def require_roles(allowed_roles: List[UserRole]):
    """
    Factory: returns a FastAPI dependency that enforces role membership.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_roles([UserRole.HR]))])
        async def admin_endpoint(): ...

    Or as a parameter dependency:
        async def endpoint(user: User = Depends(require_roles([UserRole.HR]))): ...

    Args:
        allowed_roles: List of UserRole enum values that are permitted access.

    Returns:
        An async dependency function that validates the current user's role
        and returns the User object if authorized.

    Raises:
        HTTPException 403: If the current user's role is not in allowed_roles.
    """
    async def role_guard(current_user: User = Depends(get_current_user)) -> User:
        """Inner dependency that checks the user's role against allowed roles."""
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {[r.value for r in allowed_roles]}"
            )
        return current_user
    return role_guard


# ── Convenience shorthands ──
# Use these directly as FastAPI dependencies for common access patterns.

require_hr = require_roles([UserRole.HR])
"""Only HR users can access."""

require_hr_or_exec = require_roles([UserRole.HR, UserRole.EXECUTIVE])
"""HR or Executive users can access."""

require_any = require_roles([UserRole.HR, UserRole.EMPLOYEE, UserRole.EXECUTIVE, UserRole.INTERN])
"""Any authenticated user with a valid role can access."""
