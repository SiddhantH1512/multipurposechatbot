from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.auth.jwt import create_access_token, get_current_user, verify_password
from src.database.engine import get_async_session_dep
from src.database.table_models import User


auth_router = APIRouter(prefix="/auth", tags=["auth"])

@auth_router.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_session_dep)
):
    result = await session.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/me", response_model=dict)
async def get_current_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role.value if hasattr(current_user.role, "value") else current_user.role,
        "department": current_user.department,
        "designation": current_user.designation,
        "is_active": current_user.is_active,
        "tenant_id": getattr(current_user, "tenant_id", "default")
    }