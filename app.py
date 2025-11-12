import os
from threading import Thread
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

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

embedding = None
vectorstore = None
retriever = None
llm = None

def load_background():
    """Load model and embeddings after Flask starts."""
    global embedding, vectorstore, retriever, llm
    print("⏳ Loading background components...")

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    try:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
        retriever = vectorstore.as_retriever()

        generator = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=128)
        llm = HuggingFacePipeline(pipeline=generator)

        print("✅ Background loading complete!")
    except Exception as e:
        print("❌ Background loading failed:", e)

Thread(target=load_background).start()

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
    return "\n\n".join(d.page_content for d in docs)

def get_rag_chain():
    if llm is None or retriever is None:
        raise Exception("Model or retriever still loading... please wait.")
    return (
        RunnableMap({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload_web", methods=["POST"])
def upload_web():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", message="⚠️ No file selected!")

    try:
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        docs = load_document(filepath)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        if vectorstore:
            vectorstore.add_documents(splits)
            vectorstore.persist()

        return render_template("index.html", message=f"✅ File '{filename}' uploaded successfully!")
    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {str(e)}")

@app.route("/ask", methods=["POST"])
def ask_web():
    try:
        question = request.get_json().get("question", "").strip()
        if not question:
            return jsonify({"answer": "⚠️ Please enter a question."})
        answer = get_rag_chain().invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400
        answer = get_rag_chain().invoke(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
