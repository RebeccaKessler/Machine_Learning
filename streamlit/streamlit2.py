import sqlite3

def init_db():
    conn = sqlite3.connect('user_books.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user
        ([user_id] INTEGER PRIMARY KEY, [username] text, [email] text)
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS books
        ([book_id] INTEGER PRIMARY KEY, [user_id] INTEGER, [book_title] text, [difficulty_level] text,
        FOREIGN KEY(user_id) REFERENCES user(user_id))
    ''')
    conn.commit()
    conn.close()

init_db()



def register_user(username, email):
    conn = sqlite3.connect('user_books.db')
    c = conn.cursor()
    c.execute('INSERT INTO user (username, email) VALUES (?, ?)', (username, email))
    conn.commit()
    conn.close()

def login_user(email):
    conn = sqlite3.connect('user_books.db')
    c = conn.cursor()
    c.execute('SELECT * FROM user WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

username = st.sidebar.text_input("Username")
email = st.sidebar.text_input("Email")
if st.sidebar.button("Register"):
    register_user(username, email)
    st.sidebar.success("You are registered!")

if st.sidebar.button("Login"):
    user = login_user(email)
    if user:
        st.sidebar.success("Logged in successfully!")
    else:
        st.sidebar.error("User not found.")


def save_book(user_id, book_title, difficulty_level):
    conn = sqlite3.connect('user_books.db')
    c = conn.cursor()
    c.execute('INSERT INTO books (user_id, book_title, difficulty_level) VALUES (?, ?, ?)', 
              (user_id, book_title, difficulty_level))
    conn.commit()
    conn.close()


def get_user_books(user_id):
    conn = sqlite3.connect('user_books.db')
    c = conn.cursor()
    c.execute('SELECT book_title, difficulty_level FROM books WHERE user_id = ?', (user_id,))
    books = c.fetchall()
    conn.close()
    return books

if user:  # Assuming `user` is the logged-in user tuple (user_id, username, email)
    user_books = get_user_books(user[0])
    for book in user_books:
        st.write(f"{book[0]} - Difficulty: {book[1]}")



