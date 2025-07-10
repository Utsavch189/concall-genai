from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

gemini_client = genai.GenerativeModel(
        model_name='models/gemini-2.5-flash'
    )

def ask(company_name:str,qrt:str,question:str,pdf_type:str="concall"):
    try:
        vectorstore = Chroma(
            persist_directory=f"./chroma_store/{company_name}_{pdf_type}_{qrt.replace(' ', '_')}",
            embedding_function=embedding_model
        )

        prompt_template = """
            You are a highly skilled financial analyst with deep expertise in equity research, earnings call analysis, and financial statement interpretation.

            You are given a user question and a set of relevant extracted passages from a company's earnings call or financial report.

            Your task is to:
            1. **Analyze the passages critically**, identifying signals, trends, and management insights.
            2. **Answer the user’s question directly**, citing context if relevant.
            3. Maintain a **professional tone**, use **simple language**, and **avoid generic responses**.
            4. If the answer is not in the context, reply honestly that it is not available.

            ### User Question:
            {user_question}

            ### Context (from PDF):
            {context}

            ### Answer (as a financial expert):
            """
        
        retriever = vectorstore.as_retriever(search_type="similarity", k=3)
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = prompt_template.format(user_question=question, context=context)

        response = gemini_client.generate_content(prompt)

        return response.text
    except Exception as e:
        print(e)

if __name__ == "__main__":
    res = ask(
        company_name="BATA",
        qrt="Q1FY25",
        question="explain about capex planning!"
    )
    print(res)