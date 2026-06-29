import sqlite3

DATABASE = "users.db"

def create_database():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Replaced email with username
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    print("Database schema successfully configured with usernames!")