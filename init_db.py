from database import Base, DatabaseManager
from sqlalchemy import create_engine

engine = create_engine("sqlite:///rag_system.db")
Base.metadata.create_all(engine)
print("Database initialized.")