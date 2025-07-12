from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.intent_classifier import get_doc_types
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
from calendar import month_abbr
import markdown
from datetime import datetime

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

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

def ask_question(stock, query):

    doc_types = get_doc_types(query)
    print("Suggested Doc Types : ",doc_types)

    filters = {
        "$and": [
            {"stock": {"$eq": stock}},
            {"type": {"$in": doc_types}}
        ]
    }

    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 20,
        "filter": filters
    })
    
    docs = retriever.invoke(query)

    if not docs:
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 20,
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
        You are an intelligent and professional financial analyst assistant. 
        Your role is to carefully read, synthesize, and summarize information from official company documents such as annual reports, 
        earnings call transcripts, and regulatory announcements. 
        Use the provided context to generate a precise, structured, and insight-rich response to the user's query.
    
        <b>Instructions:</b>
        - Strictly use only the provided context. Never assume, extrapolate, or fabricate information.
        - Provide a clear and well-structured answer using HTML formatting and professional financial language.
        - Highlight <b>key figures</b>, <b>strategic initiatives</b>, <b>trends</b>, <b>dates</b>, and <b>relevant document references</b> (e.g., <i>Annual Report 2024</i>, <i>Concall Q4 FY25</i>).
        - Use the following formatting rules:
          • Use <b>...</b> for all important values or keywords (e.g., <b>₹5,200 crore</b>, <b>15% YoY</b>, <b>Gen AI</b>, <b>2.1% market share</b>)
          • Use <br> for line breaks.
          • Use section titles with <b>Section Title:</b> followed by line break (e.g., <b>Revenue:</b> ... <br>)
    
        <b>If the query relates to multiple time periods:</b>
        - Preferably summarize the most recent <b>3 financial years</b> unless a specific time range is clearly requested.
        - Organize chronologically with headings like:
          <b>FY2023:</b> ... <br> <b>FY2024:</b> ... <br> <b>FY2025:</b> ...
        - Show year-over-year trends or comparisons where applicable.
    
        <b>Clarification Rules:</b>
        - If the query mentions "last year", interpret it as the <b>most recently completed financial year prior to current date</b>.
        - Do not return older years unless required for trends or if specifically asked.
    
        <b>If the query involves:</b>
        - Strategy, growth, or product development:
          Use headings like <b>Growth Strategy:</b>, <b>Innovation & Technology:</b>, <b>New Revenue Streams:</b>.
        - Market share or positioning:
          Include values like <b>2.2%</b> market share with document reference.
        - Operational metrics or financials:
          Include <b>revenue</b>, <b>profit</b>, <b>EPS</b>, <b>margins</b>, <b>TCV</b>, <b>ROCE</b>, etc., tagged clearly with fiscal year and source.
    
        <b>If the answer spans multiple document types:</b>
        - Attribute each insight inline with the document source:
          E.g., "<b>₹64,479 crore</b> revenue (<i>Concall Q4 FY25</i>)" or "<b>2.1%</b> global market share (<i>Annual Report 2024</i>)"
    
        <b>Conclusion:</b>  
        - End with a concluding paragraph in <b>Conclusion:</b> ...  
        - Recap the core insight and clearly answer the user’s question.
    
        <b>Current System Time Reference:</b>  
        The current year is {datetime.now().year}, month is {datetime.now().month}, and date is {datetime.now().day}. Use this to determine what constitutes "last year" or "current year" where applicable.
    
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
    app.run(debug=True)