import asyncio
from sqlalchemy import text
from src.auth.jwt import get_password_hash
from src.database.engine import async_engine

async def regenerate_hashes():
    # Generate fresh hash using the updated function
    hashed = get_password_hash("Test@123456")
    print(f"New hash format: {hashed[:30]}...")
    
    async with async_engine.begin() as conn:
        # Update all users
        result = await conn.execute(
            text("UPDATE users SET hashed_password = :hash"),
            {"hash": hashed}
        )
        print(f"Updated {result.rowcount} users")
        
        # Test verification
        from src.auth.jwt import verify_password
        test_user = await conn.execute(
            text("SELECT hashed_password FROM users WHERE email = 'hr@example.com'")
        )
        test_hash = test_user.scalar()
        if test_hash:
            is_valid = verify_password("Test@123456", test_hash)
            print(f"Verification test: {'✅ PASS' if is_valid else '❌ FAIL'}")

if __name__ == "__main__":
    asyncio.run(regenerate_hashes())