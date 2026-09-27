#!/usr/bin/env python3
import sqlite3
from tabulate import tabulate

DB_PATH = 'rag_system.db'

def list_tables(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    return tables

def view_table(conn, table_name, limit=20):
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    print(f"\nTable: {table_name} (showing up to {limit} rows)")
    print(tabulate(rows, headers=columns, tablefmt='grid'))

def main():
    conn = sqlite3.connect(DB_PATH)
    print(f"Connected to database: {DB_PATH}")
    tables = list_tables(conn)
    print("\nTables found:")
    for i, t in enumerate(tables, 1):
        print(f"  {i}. {t}")
    
    for t in tables:
        view_table(conn, t)
    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    main()
