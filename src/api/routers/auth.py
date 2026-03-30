from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.jwt import create_access_token, get_current_user, verify_password
from src.database.engine import get_async_session_dep
from src.database.table_models import User


auth_router = APIRouter(prefix="/auth", tags=["auth"])


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    department: str
    designation: Optional[str] = None
    tenant_id: str
    is_active: bool

@auth_router.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session_dep)
):
    result = await session.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    # ← ADD THIS CHECK
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="Account has been deactivated. Please contact HR."
        )

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


@auth_router.get("/users", response_model=dict)
async def list_users(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    """List users belonging to the same tenant as current HR user"""
    if current_user.role != "HR":
        raise HTTPException(status_code=403, detail="Only HR users can view the user list")

    tenant_id = getattr(current_user, "tenant_id", "default")

    result = await session.execute(
        text("""
            SELECT id, email, role, department, designation, tenant_id, is_active
            FROM users
            WHERE tenant_id = :tenant_id
            ORDER BY role, email
        """),
        {"tenant_id": tenant_id}
    )
    users = [dict(row) for row in result.mappings()]

    for u in users:
        if hasattr(u["role"], "value"):
            u["role"] = u["role"].value

    return {"users": users}


@auth_router.patch("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    """Deactivate a user - HR only"""
    if current_user.role != "HR":
        raise HTTPException(
            status_code=403,
            detail="Only HR users can deactivate accounts"
        )

    # Prevent HR from deactivating themselves
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot deactivate your own account"
        )

    result = await session.execute(
        text("""
            UPDATE users 
            SET is_active = false 
            WHERE id = :user_id
            RETURNING email
        """),
        {"user_id": user_id}
    )

    user = result.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await session.commit()

    return {
        "message": f"User {user[0]} has been deactivated successfully",
        "user_id": user_id
    }


@auth_router.get("/me", response_model=dict)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Return current user profile (used by Streamlit)"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role),
        "department": current_user.department,
        "designation": current_user.designation,
        "is_active": current_user.is_active,
        "tenant_id": getattr(current_user, "tenant_id", "default")
    }