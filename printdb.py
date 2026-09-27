import sqlite3

conn = sqlite3.connect('rag_system.db')
cursor = conn.cursor()
for row in cursor.execute("SELECT * FROM users"):
    print(row)
conn.close()
