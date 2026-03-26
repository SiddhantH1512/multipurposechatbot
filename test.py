# add_new_users.py
import asyncio
from sqlalchemy import text
from passlib.context import CryptContext
from src.database.engine import async_engine

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def add_new_users():
    hashed = pwd_context.hash("Test@123456")
    print(f"Using hashed password: {hashed[:40]}...")

    # Only new users (excluding existing ones like hr@company.com, employee@company.com, etc.)
    users = [
        # HR Department (30 req/min)
        ('hr.specialist@company.com', 'HR', 'HR', 'HR Specialist'),
        ('hr.recruiter@company.com', 'HR', 'HR', 'Talent Acquisition'),

        # Executive Tier (50 req/min)
        ('ceo@company.com', 'EXECUTIVE', 'Leadership', 'CEO'),
        ('cto@company.com', 'EXECUTIVE', 'Leadership', 'CTO'),
        ('cfo@company.com', 'EXECUTIVE', 'Leadership', 'CFO'),

        # Manager Tier (30 req/min)
        ('eng.manager@company.com', 'EMPLOYEE', 'Engineering', 'Engineering Manager'),
        ('sales.manager@company.com', 'EMPLOYEE', 'Sales', 'Sales Manager'),
        ('marketing.manager@company.com', 'EMPLOYEE', 'Marketing', 'Marketing Manager'),

        # Senior Engineer Tier (20 req/min)
        ('senior.dev@company.com', 'EMPLOYEE', 'Engineering', 'Senior Software Engineer'),
        ('senior.ml@company.com', 'EMPLOYEE', 'Engineering', 'Senior ML Engineer'),
        ('senior.devops@company.com', 'EMPLOYEE', 'Engineering', 'Senior DevOps Engineer'),

        # Regular Employee Tier (15 req/min)
        ('employee.eng@company.com', 'EMPLOYEE', 'Engineering', 'Software Engineer'),
        ('employee.finance@company.com', 'EMPLOYEE', 'Finance', 'Financial Analyst'),
        ('employee.sales@company.com', 'EMPLOYEE', 'Sales', 'Account Executive'),
        ('employee.marketing@company.com', 'EMPLOYEE', 'Marketing', 'Content Creator'),

        # Intern Tier (5 req/min)
        ('intern.eng@company.com', 'INTERN', 'Engineering', 'Engineering Intern'),
        ('intern.hr@company.com', 'INTERN', 'HR', 'HR Intern'),
        ('intern.sales@company.com', 'INTERN', 'Sales', 'Sales Intern'),
    ]

    async with async_engine.begin() as conn:
        for email, role, dept, designation in users:
            try:
                await conn.execute(
                    text("""
                        INSERT INTO users (email, hashed_password, role, department, designation, is_active)
                        VALUES (:email, :pwd, :role, :dept, :designation, true)
                        ON CONFLICT (email) DO NOTHING
                    """),
                    {
                        "email": email,
                        "pwd": hashed,
                        "role": role,
                        "dept": dept,
                        "designation": designation
                    }
                )
                print(f"✅ Added: {email} | Role: {role} | Dept: {dept}")
            except Exception as e:
                print(f"❌ Failed {email}: {e}")

    print("\n🎉 All new users added successfully!")
    print("Password for all users: Test@123456")

if __name__ == "__main__":
    asyncio.run(add_new_users())