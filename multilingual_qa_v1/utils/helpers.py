import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

TRACK_FILE = "ingested_files.json"
TYPE_MAP = {"annual": "annual_report", "announcements": "announcement", "concall": "concall"}

def clean_chunks(chunks):
    """
    Remove very short, boilerplate, or noisy chunks.
    """
    filtered = []
    for doc in chunks:
        text = doc.page_content.strip()
        if len(text) < 30:
            continue
        if text.lower().startswith("page"):
            continue
        if any(kw in text.lower() for kw in ["safe harbor", "disclaimer", "copyright"]):
            continue
        filtered.append(doc)
    return filtered

def load_ingested():
    if os.path.exists(TRACK_FILE):
        try:
            with open(TRACK_FILE, "r") as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {TRACK_FILE} is invalid. Reinitializing it.")
            return {}
    return {}

def extract_year(file_name):
    match = re.search(r"\d{4}", file_name)
    return int(match.group()) if match else None

def save_ingested(data):
    with open(TRACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_new_pdfs(base_dir="./reports"):
    ingested = load_ingested()
    new_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for stock in os.listdir(base_dir):
        stock_path = os.path.join(base_dir, stock)
        if not os.path.isdir(stock_path):
            continue

        ingested.setdefault(stock, {})

        for folder in os.listdir(stock_path):
            folder_path = os.path.join(stock_path, folder)
            if not os.path.isdir(folder_path):
                continue

            doc_type = TYPE_MAP.get(folder.lower(), folder.lower())
            ingested[stock].setdefault(doc_type, [])

            for file in os.listdir(folder_path):
                if not file.endswith(".pdf") or file in ingested[stock][doc_type]:
                    continue

                file_path = os.path.join(folder_path, file)
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")
                    continue

                chunks = splitter.split_documents(pages)
                chunks = clean_chunks(chunks)

                if not chunks:
                    print(f"⚠️ No usable chunks found in {file}")
                    continue

                print(f"✅ {len(chunks)} chunks from {stock} → {doc_type} → {file}")

                for chunk in chunks:
                    chunk.metadata = {
                        "stock": stock,
                        "type": doc_type,
                        "source": file,
                        "year": extract_year(file),
                    }
                    new_chunks.append(chunk)

                ingested[stock][doc_type].append(file)

    save_ingested(ingested)
    return new_chunks