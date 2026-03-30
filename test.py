import asyncio
import bcrypt
from sqlalchemy import text
from src.database.engine import async_engine

async def add_new_users():
    # Direct bcrypt hashing (reliable, no passlib issues)
    password = "Test@123456"
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))
    hashed_str = hashed.decode('utf-8')
    
    print(f"✅ Password hashed successfully with direct bcrypt")

    users = [
        # HR Department - company2
        ('hr.specialist@company2.com', 'HR', 'HR', 'HR Specialist'),
        ('hr.recruiter@company2.com', 'HR', 'HR', 'Talent Acquisition'),

        # Executive Tier
        ('ceo@company2.com', 'EXECUTIVE', 'Leadership', 'CEO'),
        ('cto@company2.com', 'EXECUTIVE', 'Leadership', 'CTO'),
        ('cfo@company2.com', 'EXECUTIVE', 'Leadership', 'CFO'),

        # Manager Tier
        ('eng.manager@company2.com', 'EMPLOYEE', 'Engineering', 'Engineering Manager'),
        ('sales.manager@company2.com', 'EMPLOYEE', 'Sales', 'Sales Manager'),
        ('marketing.manager@company2.com', 'EMPLOYEE', 'Marketing', 'Marketing Manager'),

        # Senior Engineers
        ('senior.dev@company2.com', 'EMPLOYEE', 'Engineering', 'Senior Software Engineer'),
        ('senior.ml@company2.com', 'EMPLOYEE', 'Engineering', 'Senior ML Engineer'),
        ('senior.devops@company2.com', 'EMPLOYEE', 'Engineering', 'Senior DevOps Engineer'),

        # Regular Employees
        ('employee.eng@company2.com', 'EMPLOYEE', 'Engineering', 'Software Engineer'),
        ('employee.finance@company2.com', 'EMPLOYEE', 'Finance', 'Financial Analyst'),
        ('employee.sales@company2.com', 'EMPLOYEE', 'Sales', 'Account Executive'),
        ('employee.marketing@company2.com', 'EMPLOYEE', 'Marketing', 'Content Creator'),

        # Interns
        ('intern.eng@company2.com', 'INTERN', 'Engineering', 'Engineering Intern'),
        ('intern.hr@company2.com', 'INTERN', 'HR', 'HR Intern'),
        ('intern.sales@company2.com', 'INTERN', 'Sales', 'Sales Intern'),
    ]

    async with async_engine.begin() as conn:
        for email, role, dept, designation in users:
            try:
                await conn.execute(
                    text("""
                        INSERT INTO users 
                        (email, hashed_password, role, department, designation, is_active, tenant_id)
                        VALUES (:email, :pwd, :role, :dept, :designation, true, :tenant_id)
                        ON CONFLICT (email) DO NOTHING
                    """),
                    {
                        "email": email,
                        "pwd": hashed_str,
                        "role": role,
                        "dept": dept,
                        "designation": designation,
                        "tenant_id": "company2"
                    }
                )
                print(f"✅ Added: {email} | Role: {role} | Dept: {dept} | Tenant: company2")
            except Exception as e:
                print(f"❌ Failed {email}: {e}")

    print("\n🎉 All company2 users added successfully!")
    print("Password for all users: Test@123456")

if __name__ == "__main__":
    asyncio.run(add_new_users())

# from src.database.engine import async_engine
# from sqlalchemy import text
# import asyncio

# async def assign_tenants():
#     async with async_engine.begin() as conn:
#         # Assign tenant based on email domain
#         await conn.execute(text("""
#             UPDATE users 
#             SET tenant_id = 
#                 CASE 
#                     WHEN email LIKE '%@company.com' THEN 'company'
#                     WHEN email LIKE '%@example.com' THEN 'example'
#                     ELSE 'default'
#                 END
#             WHERE tenant_id = 'default' OR tenant_id IS NULL;
#         """))
        
#         # Sync thread_metadata as well
#         await conn.execute(text("""
#             UPDATE thread_metadata 
#             SET tenant_id = 
#                 (SELECT tenant_id FROM users 
#                  WHERE users.id = thread_metadata.user_id)
#             WHERE tenant_id = 'default' OR tenant_id IS NULL;
#         """))
        
#         print("✅ Tenant assignment completed successfully!\n")

#         # Show results
#         result = await conn.execute(text("""
#             SELECT email, tenant_id, role 
#             FROM users 
#             ORDER BY email;
#         """))
        
#         print("Current Users and their Tenant IDs:")
#         print("-" * 70)
#         for row in result:
#             print(f"{row[0]:<35} →  {row[1]:<10}  (Role: {row[2]})")

# if __name__ == "__main__":
#     asyncio.run(assign_tenants())