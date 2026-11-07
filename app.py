import os
import uuid
import sqlite3
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, session
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredImageLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is missing from your .env file.")
UPLOAD_FOLDER = "uploads"
CHROMA_DIR = "chroma_store"
DB_FILE = "docanalyzer.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
prompt = ChatPromptTemplate.from_template("""
You are a professional document analyst.
Answer the question based ONLY on the context below.
If the answer is not in the documents, say exactly:
"This information is not available in the uploaded document."

Context:
{context}

Question: {question}
""")


def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    return conn


def new_conversation():
    cid = str(uuid.uuid4())
    conn = db()
    conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (cid, "New chat"))
    conn.commit()
    conn.close()
    session["conversation_id"] = cid
    return cid


def current_conversation():
    cid = session.get("conversation_id")
    if not cid:
        cid = new_conversation()
    return cid


def get_vectordb(cid):
    return Chroma(
        collection_name=f"conv_{cid.replace('-', '_')}",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR
    )


@app.route("/")
def index():
    cid = current_conversation()
    conn = db()
    conversations = conn.execute(
        "SELECT id, title FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    messages = conn.execute(
        "SELECT question, answer FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,)
    ).fetchall()
    documents = conn.execute(
        "SELECT filename FROM documents WHERE conversation_id=? ORDER BY id",
        (cid,)
    ).fetchall()
    conn.close()
    return render_template("index.html",
                           conversations=conversations,
                           messages=messages,
                           documents=documents,
                           active_id=cid)

@app.route("/new_chat")
def new_chat():
    new_conversation()
    return redirect(url_for("index"))


@app.route("/chat/<cid>")
def open_chat(cid):
    session["conversation_id"] = cid
    return redirect(url_for("index"))


@app.route("/clear_history")
def clear_history():
    conn = db()
    conn.execute("DELETE FROM messages")
    conn.execute("DELETE FROM documents")
    conn.execute("DELETE FROM conversations")
    conn.commit()
    conn.close()
    session.pop("conversation_id", None) #we use this pop to remove the conversation_id from the session, effectively clearing the current conversation.
    return redirect(url_for("index"))


@app.route("/upload", methods=["POST"])
def upload():
    cid = current_conversation()
    file = request.files.get("file")
    
    if not file or file.filename == "":
        return redirect(url_for("index"))
    if not file.filename.lower().endswith((".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png")):
        return "Please upload a valid file.", 400
    path = os.path.join(UPLOAD_FOLDER, f"{cid}_{file.filename}")
    file.save(path)
    if os.path.getsize(path) == 0:
        return "Uploaded file is empty.", 400
    try:
        if file.filename.lower().endswith(".pdf"):
            chunks = splitter.split_documents(PyPDFLoader(path).load())
        elif file.filename.lower().endswith(".docx"):
            chunks = splitter.split_documents(Docx2txtLoader(path).load())
        elif file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            chunks = splitter.split_documents(UnstructuredImageLoader(path).load())
        elif file.filename.lower().endswith(".doc"):
            chunks = splitter.split_documents(UnstructuredWordDocumentLoader(path).load())
        elif file.filename.lower().endswith(".txt"):
            chunks = splitter.split_documents(TextLoader(path).load())
        else:
            return "Unsupported file type.", 400
    except Exception as e:
        return f"Error processing file: {str(e)}", 500

    for chunk in chunks:
        chunk.metadata["source_file"] = file.filename
    get_vectordb(cid).add_documents(chunks)

    conn = db()
    conn.execute("INSERT INTO documents (conversation_id, filename) VALUES (?, ?)",
                 (cid, file.filename))
    conn.execute("UPDATE conversations SET title=? WHERE id=? AND title='New chat'",
                 (file.filename, cid))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))


@app.route("/ask", methods=["POST"])
def ask():
    cid = current_conversation()
    question = request.form.get("question", "").strip()
    if not question:
        return redirect(url_for("index"))

    results = get_vectordb(cid).similarity_search(question, k=4)
    if not results:
        answer = "Please upload a valid file first."
    else:
        context = "\n\n".join(r.page_content for r in results)
        answer = (prompt | llm).invoke(
            {"context": context, "question": question}
        ).content

    conn = db()
    conn.execute("INSERT INTO messages (conversation_id, question, answer) VALUES (?, ?, ?)",
                 (cid, question, answer))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)