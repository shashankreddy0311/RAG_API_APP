import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

from utils.document_loader import load_document

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PERSIST_DIR = "chroma_store"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
retriever = vectorstore.as_retriever()

generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=128)
llm = HuggingFacePipeline(pipeline=generator)
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

rag_chain = (
    RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# ---------- Web interface ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/upload_web", methods=["POST"])
def upload_web():
    """Handle file uploads from the web form"""
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
        print("Extracted text sample:", splits[0].page_content[:500])
        vectorstore.add_documents(splits)
        vectorstore.persist()

        return render_template("index.html", message=f"✅ File '{filename}' uploaded successfully!")
    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {str(e)}")

@app.route("/ask", methods=["POST"])
def ask_web():
    """Handle question submissions from the web frontend (AJAX)."""
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "⚠️ Please enter a question."}), 400

    try:
        print(f"\n❓ User asked: {question}")
        retrieved_docs = retriever.invoke(question)
        if not retrieved_docs:
            print("⚠️ No relevant documents found.")
            return jsonify({"answer": "No relevant information found in your uploaded file."})
        
        print("\n📚 Retrieved context sample:\n", retrieved_docs[0].page_content[:300])
        answer = rag_chain.invoke(question)
        print("\n💬 Model Answer:\n", answer)

        if not answer.strip():
            return jsonify({"answer": "⚠️ Model returned no answer. Try rephrasing your question."})
        
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"❌ Error during answering: {e}")
        return jsonify({"answer": f"Error: {str(e)}"}), 500


# ---------- Existing API endpoints ----------
@app.route("/upload_file", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    docs = load_document(filepath)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vectorstore.add_documents(splits)
    vectorstore.persist()
    return jsonify({"message": f"File '{filename}' uploaded successfully.", "chunks_added": len(splits)}), 200

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question")
    answer = rag_chain.invoke(question)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


