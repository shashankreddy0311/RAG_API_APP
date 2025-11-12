import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if not docs or not any(d.page_content.strip() for d in docs):
                    raise ValueError("Empty PDF text")
            except Exception:
                # fallback: OCR using pytesseract
                print("⚙️ Running OCR on PDF...")
                images = convert_from_path(file_path)
                text = ""
                for img in images:
                    text += pytesseract.image_to_string(img)
                docs = [type("Doc", (object,), {"page_content": text})()]
        elif ext == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
        elif ext == ".pptx":
            loader = UnstructuredPowerPointLoader(file_path)
            docs = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        docs = [d for d in docs if getattr(d, "page_content", "").strip()]
        if not docs:
            raise ValueError("No readable text found in file.")
        return docs

    except Exception as e:
        raise RuntimeError(f"Document loading failed: {e}")
