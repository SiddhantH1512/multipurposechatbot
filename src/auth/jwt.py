from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.table_models import User
from src.database.engine import get_async_session, get_async_session_dep
from src.config import Config
import bcrypt  # Import bcrypt directly

SECRET_KEY = Config.SECRET_KEY

# Create a custom CryptContext that works with bcrypt 4.x
class FixedCryptContext(CryptContext):
    """Custom CryptContext that handles bcrypt 4.x correctly"""
    
    def _get_record(self, scheme, category=None):
        """Override to handle bcrypt 4.x"""
        try:
            return super()._get_record(scheme, category)
        except Exception as e:
            if "bcrypt" in str(e):
                # Fallback: use direct bcrypt for verification
                return None
            raise

# Create password context with bcrypt
pwd_context = CryptContext(
    schemes=["bcrypt"],
    bcrypt__default_rounds=12,
    deprecated="auto"
)

# Add direct bcrypt fallback functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compare plain password against stored hash with fallback"""
    try:
        # Try passlib first
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # If passlib fails, try direct bcrypt
        try:
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except:
            # If both fail, log and return False
            print(f"Password verification failed for hash: {hashed_password[:20]}...")
            return False

def get_password_hash(password: str) -> str:
    """Hash a new password before saving"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create signed JWT"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Rest of the file remains the same...
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: AsyncSession = Depends(get_async_session_dep)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    return user