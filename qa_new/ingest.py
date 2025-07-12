from utils.helpers import load_new_pdfs
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
docs = load_new_pdfs()

if not docs:
    print("✅ No new files to embed.")
else:
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    vectorstore.add_documents(docs)
    print(f"✅ Added {len(docs)} new chunks from new PDFs.")
