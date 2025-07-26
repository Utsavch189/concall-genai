import fitz  # PyMuPDF
from PIL import Image
import os
import io

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    content = []
    for i, page in enumerate(doc):
        text = page.get_text()
        images = page.get_images(full=True)
        image_paths = []
        for img_index, img in enumerate(images):
            base_image = doc.extract_image(img[0])
            image_bytes = base_image["image"]
            img_ext = base_image["ext"]
            image_path = f"images/temp_img_{i}_{img_index}.{img_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_paths.append(image_path)
        content.append({"text": text, "images": image_paths})
    return content

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch


def caption_image(image_path: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    image = Image.open(image_path).convert("RGB")
    prompt = "Provide a short caption for the image."
    
    response = model.generate_content([prompt, image])
    return response.text.strip()


from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_vectorstore(parsed_chunks):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for entry in parsed_chunks:
        text = entry["text"]
        captions = " ".join(entry["captions"])
        merged_text = f"{text}\n\nImage Context: {captions}"
        for chunk in splitter.split_text(merged_text):
            docs.append(Document(page_content=chunk))
    vectorstore = Chroma.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), persist_directory="./data/chroma_db")
    vectorstore.persist()

def ingest_pdf(pdf_path):
    raw_content = extract_text_and_images(pdf_path)
    processed = []
    for item in raw_content:
        captions = [caption_image(img) for img in item["images"]]
        processed.append({"text": item["text"], "captions": captions})
    build_vectorstore(processed)

def query_system(query):
    chunks = retrieve_chunks(query)
    prompt = build_prompt(chunks, query)
    return generate_answer(prompt)

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def retrieve_chunks(query, top_k=4):
    db = Chroma(persist_directory="./chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    results = db.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]

def build_prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:
"""



model = genai.GenerativeModel(model_name="gemini-2.5-flash")


def generate_answer(prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    ingest_pdf("reports/TCS/annual/AnnualReport2023.pdf")
    # print(query_system("What were the key financial highlights of FY 2024?"))