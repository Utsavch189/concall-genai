from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.intent_classifier import get_doc_types
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
from calendar import month_abbr
import markdown

def convert_markdown_bold_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

def parse_filename(file_name):
    """
    Parses filenames like 'M6Y2025D20.pdf' or 'Q4_2025.pdf' into human-readable labels.
    """
    base_name = file_name.replace(".pdf", "")  # remove extension if needed

    # Match format: M6Y2025D20
    match_month = re.search(r"M(\d{1,2})Y(\d{4})D(\d{1,2})", base_name, re.IGNORECASE)
    if match_month:
        month_num, year, day = match_month.groups()
        try:
            month_name = month_abbr[int(month_num)]
            return f"{month_name} {int(day)}, {year}"
        except IndexError:
            pass  # Invalid month, fallback to filename

    # Match format: Q4_2025 or Q4-2025 or Q4 2025
    match_quarter = re.search(r"Q([1-4])[_\- ]?(\d{4})", base_name, re.IGNORECASE)
    if match_quarter:
        quarter, year = match_quarter.groups()
        return f"Q{quarter} FY {year}"

    return base_name
    
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_question(stock, query):

    doc_types = get_doc_types(query)
    print("Suggested Doc Types : ",doc_types)

    filters = {
        "$and": [
            {"stock": {"$eq": stock}},
            {"type": {"$in": doc_types}}
        ]
    }
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 10,
        "filter": filters
    })
    
    docs = retriever.invoke(query)

    if not docs:
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 10,
            "filter": {
                "stock": stock
            }
        })
        docs = retriever.invoke(query)

    sources = list({
        f"{d.metadata['type'].replace('_', ' ').title()} - {parse_filename(d.metadata['source'])}"
        for d in docs
    })
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
        You are a smart financial analyst assistant. Based on the provided context extracted from official company documents 
        (e.g., annual reports, earnings call transcripts, announcements), answer the user’s question with precision, structure, and financial clarity.

        <b>Instructions:</b>
        - Use only the information from the provided context. Do not assume or fabricate.
        - Present the answer in clear paragraphs or structured bullet points.
        - Where applicable, include <b>key financial metrics, dates, and document references</b> (e.g., Concall Q4 FY25, Annual Report 2024).
        - Use <b>bullet points</b> or sectioned formatting when summarizing multiple insights.
        - When using subheadings, enclose the title in <b>&lt;b&gt;</b> tags and separate each section with a line break (\n).  
          Example:  
          <b>Revenue:</b> ... \n <b>Strategy:</b> ...
        - Use <b>&lt;b&gt;</b> tags to highlight all important figures or values such as <b>₹200 crore</b>, <b>40%</b>, <b>15% YoY</b>, <b>₹64,479</b>, <b>AI</b>, etc.
        - If data is spread across multiple years or documents, highlight patterns or trends concisely.
        - Be professional, concise, and avoid subjective or speculative language.
        - Do not use Markdown (e.g., **bold**). Use HTML tags like <b>...</b> for all emphasis.
        - Bold important phrases like <b>₹200 crore</b>, <b>Gen AI</b>, etc. using HTML tags
        - For line beaks use \n.

        <b>Conclusion:</b>  
        - Always end with a summarizing conclusion in a single bullet or short paragraph enclosed in <b>Conclusion:</b> …  
          Focus on what the insights imply or confirm regarding the user's question.

        ---

        <b>Company:</b> {stock}  
        <b>Question:</b> "{query}"

        ---

        <b>Context:</b>  
        {context}

        ---

        <b>Answer:</b>
        """

    response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)

    content = response.text.strip()
    cleaned_output = convert_markdown_bold_to_html(content)

    return {
        "stock": stock,
        "question": query,
        "sources": sources,
        "document_count": len(sources),
        "reply": markdown.markdown(cleaned_output)
    }


from flask import Flask,request,jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app,origins="*")

@app.post('/chat')
def chat():
    query = request.json.get('query')
    res = ask_question(
        stock='TCS',
        query=query
    )
    # print(res.get('reply'))
    return jsonify(res)


if __name__ == "__main__":
    # res = ask_question(
    #     stock='TCS',
    #     query="What has the company been doing to grow? Has the company created any new revenue streams over the last 3 years? Has it launched any new products or innovations?"
    # )
    # print(res)
    app.run()