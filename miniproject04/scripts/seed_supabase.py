"""
Supabase Seeding Script.

Bypasses the Supabase UI file size limits by connecting directly 
to the database and executing the raw SQL dump.
"""

import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

def seed_database():
    """Seed the MediCore PostgreSQL database from the medicore_data.sql dump file."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=project_root / ".env")
    
    db_url = os.getenv("DB_CONNECTION_STRING")
    sql_file = project_root / "data" / "medicore_data.sql"

    if not db_url:
        print("❌ Error: DB_CONNECTION_STRING not found in .env")
        return

    if not sql_file.exists():
        print(f"❌ Error: Could not find {sql_file.relative_to(project_root)}")
        return

    print("Reading SQL dump...")
    with open(sql_file, "r", encoding="utf-8") as f:
        sql_script = f.read()

    print("Connecting to Supabase and executing...")
    try:
        # Connect directly using psycopg2
        conn = psycopg2.connect(db_url)
        conn.autocommit = True  # Ensures the execution commits immediately
        
        with conn.cursor() as cur:
            cur.execute(sql_script)
            
        print("✅ Success! Your tables have been created and seeded in Supabase.")
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        if 'conn' in locals() and conn is not None:
            conn.close()

if __name__ == "__main__":
    seed_database()