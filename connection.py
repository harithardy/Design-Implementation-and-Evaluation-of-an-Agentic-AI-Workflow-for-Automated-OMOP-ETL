import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load .env from the project root (one level up from /src)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Read credentials from .env
DB_SERVER = os.getenv("DB_SERVER")
DB_NAME   = os.getenv("DB_NAME")
DB_DRIVER = os.getenv("DB_DRIVER")  # Already URL-encoded in .env (spaces as +)

# Build connection string
connection_string = (
    f"mssql+pyodbc://{DB_SERVER}/{DB_NAME}"
    f"?driver={DB_DRIVER}"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

# Create engine
engine = create_engine(connection_string, isolation_level="AUTOCOMMIT")

def test_connection():
    """Quick check that the connection works."""
    with engine.connect() as conn:
        version = conn.execute(text("SELECT @@VERSION")).fetchone()
        print("Connected successfully!")
        print(f"   Server: {DB_SERVER}")
        print(f"   Database: {DB_NAME}")
        print(f"   SQL Server version: {version[0].splitlines()[0]}")

if __name__ == "__main__":
    test_connection()