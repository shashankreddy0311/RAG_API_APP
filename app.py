import os
from threading import Thread
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.document_loader import load_document

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PERSIST_DIR = "chroma_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# ---- Initialize embeddings and vectorstore ----
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
retriever = vectorstore.as_retriever()

# ---- Lazy model loading setup ----
llm = None  # Placeholder for LLM

def load_model_background():
    """Load the text generation model in a background thread."""
    global llm
    from transformers import pipeline
    from langchain_community.llms import HuggingFacePipeline

    print("⏳ Loading model in background (Flan-T5-small)...")
    try:
        generator = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=128)
        llm = HuggingFacePipeline(pipeline=generator)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print("❌ Failed to load model:", e)

# Start loading the model without blocking Flask startup
Thread(target=load_model_background).start()

# ---- Prompt Template ----
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful AI assistant. Use the following context to answer the question clearly and concisely.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "If the answer is not in the context, say 'The answer is not available in the provided document.'\n\n"
        "Answer:"
    )
)

def format_docs(docs):
    """Join retrieved documents as text."""
    return "\n\n".join(d.page_content for d in docs)

def get_rag_chain():
    """Dynamically build the RAG chain after model loads."""
    if llm is None:
        raise Exception("Model is still loading... please wait 20–30 seconds.")
    return (
        RunnableMap({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

# ---------- Web Interface ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload_web", methods=["POST"])
def upload_web():
    """Handle file uploads from the web form."""
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", message="⚠️ No file selected!")
    try:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        docs = load_document(filepath)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        if not splits:
            return render_template("index.html", message="⚠️ No text could be extracted from the file.")
        vectorstore.add_documents(splits)
        vectorstore.persist()

        return render_template("index.html", message=f"✅ File '{filename}' uploaded successfully!")
    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {str(e)}")

@app.route("/ask", methods=["POST"])
def ask_web():
    """Handle question submissions from the web form."""
    try:
        question = request.get_json().get("question", "").strip()
        if not question:
            return jsonify({"answer": "⚠️ Please enter a question."})
        answer = get_rag_chain().invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})

# ---------- API Endpoints ----------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    """API: upload file via REST request."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    docs = load_document(filepath)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    if not splits:
        return jsonify({"error": "No content extracted from the document."}), 400
    vectorstore.add_documents(splits)
    vectorstore.persist()

    return jsonify({"message": f"File '{filename}' uploaded successfully.", "chunks_added": len(splits)}), 200

@app.route("/query", methods=["POST"])
def query():
    """API: answer query via JSON request."""
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400
        answer = get_rag_chain().invoke(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
