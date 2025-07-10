from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

gemini_client = genai.GenerativeModel(
        model_name='models/gemini-2.5-flash'
    )

def load_concall_pdf(pdf_path: str):
    try:
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        print("Error loading PDF:", e)


def split_pdf_into_chunks(pdf_path: str):
    try:
        documents = load_concall_pdf(pdf_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(documents)
    except Exception as e:
        print("Error splitting PDF:", e)

def train(company_name:str,qrt:str,pdf_path:str,pdf_type:str="concall"):
    try:
        print(f"Training on: {company_name} - {qrt}")
        chunks = split_pdf_into_chunks(pdf_path)

        if not chunks:
            print("No chunks to process.")
            return

        for doc in chunks:
            doc.metadata["company"] = company_name
            doc.metadata["quarter"] = qrt
        
        persist_dir = f"./chroma_store/{company_name}_{pdf_type}_{qrt.replace(' ', '_')}"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        print(f"Training complete. Data stored at: {persist_dir}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    train(
        company_name="BLUESTAR",
        qrt="Q4FY25",
        pdf_path="blue star.pdf"
    )