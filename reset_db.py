#!/usr/bin/env python3
import os
from database import create_all_tables

def reset_database(db_path='rag_system.db'):
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted old database: {db_path}")
    create_all_tables(f"sqlite:///{db_path}")
    print(f"Created new database: {db_path}")

if __name__ == "__main__":
    reset_database()
