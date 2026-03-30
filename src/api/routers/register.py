from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.jwt import get_current_user, get_password_hash
from src.database.engine import get_async_session_dep
from src.database.table_models import User

register_router = APIRouter(prefix="/auth", tags=["auth"])


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    role: str = "EMPLOYEE"
    department: str
    designation: str | None = None
    tenant_id: str = "default"   # HR can specify tenant


@register_router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegister,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session_dep)
):
    """Register a new user - Only HR users can register new employees"""
    
    # Only HR can register new users
    if current_user.role != "HR":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only HR users can register new employees"
        )

    # Check if user already exists
    result = await session.execute(
        text("SELECT id FROM users WHERE email = :email"),
        {"email": user_data.email}
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists"
        )

    # Validate role
    valid_roles = {"HR", "EXECUTIVE", "EMPLOYEE", "INTERN"}
    if user_data.role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    if not user_data.tenant_id or len(user_data.tenant_id.strip()) == 0:
        user_data.tenant_id = "default"

    # Hash password
    hashed_password = get_password_hash(user_data.password)

    # Insert new user
    await session.execute(
        text("""
            INSERT INTO users 
            (email, hashed_password, role, department, designation, tenant_id, is_active)
            VALUES 
            (:email, :hashed_password, :role, :department, :designation, :tenant_id, true)
        """),
        {
            "email": user_data.email,
            "hashed_password": hashed_password,
            "role": user_data.role,
            "department": user_data.department,
            "designation": user_data.designation,
            "tenant_id": user_data.tenant_id
        }
    )
    await session.commit()

    return {
        "message": "User registered successfully",
        "email": user_data.email,
        "role": user_data.role,
        "department": user_data.department,
        "tenant_id": user_data.tenant_id
    }