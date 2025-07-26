from utils.helpers import load_new_pdfs
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def chunk_docs_list(docs, chunk_size=2000):
    return [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

try:

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = chunk_docs_list(load_new_pdfs())

    # print(len(docs))
    # print()
    # print(docs)

    if not docs:
        print("✅ No new files to embed.")
    else:
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
        for d in docs:
            vectorstore.add_documents(d)
            print(f"✅ Added {len(d)} new chunks from new PDFs.")
        print(f"Ultimate docs length {len(docs)}")

except Exception as e:
    print(e)
