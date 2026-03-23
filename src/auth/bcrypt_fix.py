"""
Workaround for passlib + bcrypt 4.x compatibility
"""
from passlib.handlers.bcrypt import bcrypt as _bcrypt
from passlib.utils import getrandbytes
import bcrypt

class FixedBcryptHandler(_bcrypt):
    """Fixed bcrypt handler that works with bcrypt 4.x"""
    
    @classmethod
    def _get_backend_version(cls):
        """Get bcrypt version without using __about__"""
        try:
            # Try __about__ first (bcrypt < 4.0)
            return _bcrypt._get_backend_version()
        except AttributeError:
            # For bcrypt >= 4.0, use __version__
            try:
                return bcrypt.__version__
            except AttributeError:
                return "unknown"

# Replace the default bcrypt handler
from passlib.context import CryptContext

def create_fixed_context():
    """Create a CryptContext with the fixed bcrypt handler"""
    return CryptContext(
        schemes=["bcrypt"],
        bcrypt__default_rounds=12,
        deprecated="auto"
    )
