import os
import requests
from flask import Flask, request, jsonify,render_template
from dotenv import load_dotenv
from flask_cors import CORS
from openai import OpenAI
import io
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.intent_classifier import get_relevant_sources
import google.generativeai as genai
import re
import markdown
from utils.text_translation import translate_text_v1,translate_text_v2,language_codes
from utils.model_costs import cost_gemini_25_flash,cost_gpt4o

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEy'))
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
        result = vectorstore.similarity_search(query, k=5, filter=f)
        print(result)
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
        You are a highly intelligent, detail-oriented, and trustworthy financial analyst assistant.  
        Your sole responsibility is to generate **clear**, **concise**, and **data-driven** responses strictly using the <b>verified context</b> extracted from official company documents (e.g., Annual Reports, Earnings Calls, Regulatory Filings).

        <hr>
        <b>🧾 Company:</b> {stock}<br>
        <b>📌 Query:</b> "{query}"<br><br>
        <b>📚 Context:</b><br><br>
        {context}<br><br>

        <hr>
        <h3>🛠️ Answer Construction Guidelines:</h3>

        <b>1. Adherence to Context:</b><br>
        <ul>
          <li>Use only the information provided in the context above. <b>Do not infer, assume, or fabricate</b> any information.</li>
          <li>If the context does not contain relevant data, respond exactly with: <i>No relevant data found in the provided documents.</i></li>
        </ul><br>

        <b>2. Formatting & Visual Clarity:</b><br>
        <ul>
          <li>Use HTML formatting to ensure structured and readable output.</li>
          <li>Highlight important metrics using <b>bold</b> (e.g., <b>₹8,520 crore</b>, <b>14% YoY growth</b>, <b>Net Profit</b>).</li>
          <li>Use line breaks (&lt;br&gt;) between facts and bullet points (&lt;ul&gt;&lt;li&gt;...&lt;/li&gt;&lt;/ul&gt;) or tables for organized data.</li>
          <li>Always include appropriate units and currency (e.g., <b>₹ crore</b>, <b>million USD</b>, <b>%</b>).</li>
        </ul><br>

        <b>3. Handling Specific Query Types:</b><br>

        <ul>
          <li><b>A. Multi-Year Financial Metrics (e.g., Revenue, PAT, EBITDA):</b>
            <ul>
              <li>Present each year’s data clearly: <br>FY2022: ₹X crore<br>FY2023: ₹Y crore<br>FY2024: ₹Z crore</li>
              <li>If a specific year is missing, clearly state: "FY20XX data not available."</li>
              <li>Use <b>tables</b> when showing year-wise comparisons for 2+ metrics.</li>
            </ul>
          </li><br>

          <li><b>B. Trend or Comparative Analysis (e.g., margins, cost ratios, segment sales):</b>
            <ul>
              <li>State the direction and magnitude of change with phrases like:
                <ul>
                  <li><b>"increased from ₹X crore to ₹Y crore"</b></li>
                  <li><b>"declined by 12% YoY"</b></li>
                  <li><b>"remained steady at ₹Z crore"</b></li>
                </ul>
              </li>
              <li>Back every trend statement with supporting data from the context.</li>
            </ul>
          </li><br>

          <li><b>C. Strategic, Operational, or Shareholder Insights:</b>
            <ul>
              <li>Use appropriate section headers such as:<br>
                <ul>
                  <li><h3>📊 Business Model:</h3></li>
                  <li><h3>🚀 Strategic Initiatives:</h3></li>
                  <li><h3>🏭 Operational Performance:</h3></li>
                  <li><h3>📈 Growth Drivers / Risks:</h3></li>
                </ul>
              </li>
              <li>Deliver well-structured factual summaries under each section, clearly based on the provided context.</li>
            </ul>
          </li>
        </ul><br>

        <b>4. Tone & Language:</b><br>
        <ul>
          <li>Maintain a formal, objective, and analytical tone.</li>
          <li>Use precise, non-speculative language. Avoid filler words and unnecessary jargon.</li>
          <li>Ensure every sentence directly reflects data or insights from the context.</li>
        </ul><br>

        <b>5. Conclusion:</b><br>
        <ul>
          <li>End with a concise 2–3 line summary highlighting the most significant finding(s) relevant to the query.</li>
        </ul><br>

        <hr>
        <b>🧠 Final Answer:</b><br>
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

    used_sources = list(set(used_sources))

    return {
        "stock": stock,
        "question": query,
        "sources": used_sources,
        "document_count": len(used_sources),
        "reply": markdown.markdown(cleaned_html),
        "token_usage_for_rag": {
            "doc_type_tokens": doc_type_tokens,
            "answer_prompt_tokens": prompt_tokens,
            "answer_response_tokens": response_tokens,
            "total_tokens": total_tokens
        }
    }

@app.route("/",methods=["GET"])
def home():
    return render_template("index2.html")

@app.route("/translate", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    audio = ("speech.webm", io.BytesIO(audio_bytes), "audio/webm")

    translate = client.audio.translations.create(
        model="whisper-1",
        file=audio
    )
    print(translate)
    return jsonify({"text":translate.text})

@app.post('/chat/<string:lang>')
def chat(lang:str):
    query = request.json.get('query')
    
    translate = translate_text_v2(query)
    trans_txt = translate.get('translated_text')
    print(translate)
    
    res = ask_question(
        stock='TCS',
        query=trans_txt
    )

    inp_tokens = res["token_usage_for_rag"]["answer_prompt_tokens"]+res["token_usage_for_rag"]["doc_type_tokens"]["prompt_tokens"]
    out_tokens = res["token_usage_for_rag"]["answer_response_tokens"]+res["token_usage_for_rag"]["doc_type_tokens"]["response_tokens"]

    if lang!='en':
        translation = translate_text_v1(res['reply'],lang)
        res['reply'] = translation['translated_text']
        res['token_usage_for_translation'] = translation['token_usage']
        res["reply"] = translation["translated_text"]
        inp_tokens += translation["token_usage"]["input_token"]
        out_tokens += translation["token_usage"]["output_token"]

    res["rag_price_usd"] = cost_gemini_25_flash(
        inp_tokens,
        out_tokens
        )
    res["translate_price_usd"] = cost_gpt4o(translate["token_usage"]["input_token"],translate["token_usage"]["output_token"])

    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True,port=8000,host="0.0.0.0")