from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.intent_classifier import get_relevant_sources
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import markdown
from datetime import datetime

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_fn = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_fn)

def count_tokens(model, text):
    return model.count_tokens(text).total_tokens

def parse_filename(file_name):
    # Convert M6Y2025D20.pdf -> Jun 20, 2025 or Q4_2025.pdf -> Q4 FY 2025
    base = file_name.replace(".pdf", "")
    match_month = re.search(r"M(\d{1,2})Y(\d{4})D(\d{1,2})", base, re.IGNORECASE)
    if match_month:
        from calendar import month_abbr
        month_num, year, day = match_month.groups()
        return f"{month_abbr[int(month_num)]} {int(day)}, {year}"
    match_quarter = re.search(r"Q([1-4])[_\- ]?(\d{4})", base, re.IGNORECASE)
    if match_quarter:
        quarter, year = match_quarter.groups()
        return f"Q{quarter} FY {year}"
    return base


def convert_markdown_bold_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)


def ask_question(stock: str, query: str):
    # Step 1: Get all metadata for the stock
    all_docs = vectorstore._collection.get(include=["metadatas"])
    stock_metadatas = [
        m for m in all_docs["metadatas"]
        if m.get("stock", "").lower() == stock.lower()
    ]

    if not stock_metadatas:
        return {
            "stock": stock,
            "question": query,
            "sources": [],
            "document_count": 0,
            "reply": "<b>No documents found for this company.</b>"
        }

    # Step 2: Ask Gemini for relevant sources
    relevant_labels,doc_type_tokens = get_relevant_sources(query, stock_metadatas)  # returns list like ["Annual Report - Q4_2025.pdf", ...]
    print("Releavent Reports : ",relevant_labels)
    # Step 3: Extract (type, source) filters from labels
    filters = []
    for label in relevant_labels:
        try:
            type_str, source_str = label.split(" - ", 1)
            doc_type = type_str.strip().lower().replace(" ", "_")
            filters.append(
                {
                    "$and": [
                        {"stock": {"$eq": stock}},
                        {"type": {"$eq": doc_type}},
                        {"source": {"$eq": source_str}}
                    ]
                }
            )
        except ValueError:
            continue 
    print("Filters : ",filters)
    if not filters:
        return {
            "stock": stock,
            "question": query,
            "sources": [],
            "document_count": 0,
            "reply": "<b>No relevant documents selected by Gemini.</b>"
        }

    # Step 4: Collect context from matching docs
    context = ""
    used_sources = []

    for f in filters:
        # Chroma allows only one filter at a time in similarity_search
        print("Filter : ",f)
        result = vectorstore.similarity_search(query, k=5, filter=f)
        print("Result : ",result)
        print("----")
        if result:
            for doc in result:
                context += doc.page_content + "\n\n"
                label = f"{doc.metadata['type'].replace('_', ' ').title()} - {parse_filename(doc.metadata['source'])}"
                used_sources.append(label)
    
    if not context:
        return {
            "stock": stock,
            "question": query,
            "sources": relevant_labels,
            "document_count": 0,
            "reply": "<b>No matching content found in the selected documents.</b>"
        }

    prompt = f"""
        You are a smart, structured, and highly reliable financial analyst assistant. 
        Your primary goal is to provide clear, concise, and accurate answers to financial 
        queries based strictly on the provided context from official company documents (e.g., annual reports, earnings call transcripts, announcements).

        Company: {stock}

        Question: "{query}"

        Context:

        {context}

        Instructions for Generating the Answer:

        Adherence to Context:

        Use only the provided context. Never infer, assume, or generate information not explicitly present.

        If the context contains no relevant data for the query, respond directly: No relevant data found in the provided documents.

        Formatting and Presentation:

        Use HTML formatting for enhanced readability.

        Highlight key figures, financial metrics, and important facts using <b>...</b> (e.g., ₹8,520 crore, 14% YoY growth, Net Profit After Tax).

        Use <br> for line breaks within paragraphs or for spacing out bullet points.

        Employ bullet points (<ul><li>...</li></ul>) or tabular formatting (<table>...</table>) for lists, year-wise data, or comparisons to ensure clarity and conciseness.

        Always include units and currency where applicable (e.g., ₹ crore, %, million USD).

        Handling Specific Query Types:

        A. Year-over-Year (YoY) or Multi-Year Data (e.g., "What was the revenue over the past three years?"):

        Organize each metric clearly year-wise.

        Example structure:

        FY2023: [Metric Value]

        FY2024: [Metric Value]

        FY2025: [Metric Value]

        If data for a specific year is missing, state it explicitly (e.g., "FY2024 data not available").

        For growth percentages or financial metrics (Revenue, PAT, EBITDA, etc.), use compact bullet points or a clear table.

        B. Trend or Comparison Queries (e.g., "Describe the trend in gross profit margins," "Compare sales across segments"):

        Clearly identify increases, decreases, or stable patterns across periods or between categories.

        Use precise phrases such as:

        "grew by [X]%"

        "declined to [Value]"

        "increased from [Value A] to [Value B]"

        "remained stable at [Value]"

        "outperformed/underperformed"

        Quantify trends with specific numbers and percentages from the context.

        C. Summaries, Innovations, Strategy, Operations, or Shareholder Queries (e.g., "Summarize the business model," "What are the key strategic initiatives?"):

        Organize the answer using clear HTML headings (<h3>...</h3>) or bolded labels.

        Examples of headings/labels:

        Business Model: ...

        Strategic Initiatives: ...

        Shareholding Pattern: ...

        Product Launches: ...

        Operational Highlights: ...

        Provide a concise summary of the relevant information under each heading.

        Language and Tone:

        Maintain a professional, objective, and factual tone.

        Use clear, straightforward language. Avoid jargon where simpler terms suffice.

        Be direct and avoid conversational fillers or overly flowery language.

        Prioritize clarity and precision in all statements.

        Conclusion:

        Conclude the answer with a brief, insightful summary (2-3 lines) of the key findings, overall trend, or the most significant piece of information derived from the provided context relevant to the query.

        Answer:
        """

    model = genai.GenerativeModel("models/gemini-2.5-flash")
    response = model.generate_content(prompt)
    final_answer = response.text.strip()
    cleaned_html = convert_markdown_bold_to_html(final_answer)

    total_prompt_tokens = 0
    total_response_tokens = 0

    prompt_tokens = count_tokens(model, prompt)
    total_prompt_tokens += prompt_tokens

    response_tokens = count_tokens(model, response.text.strip())
    total_response_tokens += response_tokens

    total_tokens = doc_type_tokens["total_tokens"] + prompt_tokens + response_tokens

    return {
        "stock": stock,
        "question": query,
        "sources": used_sources,
        "document_count": len(used_sources),
        "reply": markdown.markdown(cleaned_html),
        "token_usage": {
            "doc_type_tokens": doc_type_tokens,
            "answer_prompt_tokens": prompt_tokens,
            "answer_response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
    }

from flask import Flask,request,jsonify,render_template
from flask_cors import CORS

app = Flask(__name__)

CORS(app,origins="*")

@app.get("/")
def chat_page():
    return render_template("chat.html")

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
    app.run(debug=True,host="0.0.0.0")