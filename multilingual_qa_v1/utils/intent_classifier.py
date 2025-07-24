import google.generativeai as genai
from datetime import datetime
from collections import defaultdict
import os

def count_tokens(model, text):
    return model.count_tokens(text).total_tokens

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_relevant_sources(query: str, all_metadata: list[dict]) -> list[str]:

    doc_labels = [
        f"{m['type'].replace('_', ' ').title()} - {m['source']}"
        for m in all_metadata
    ]

    current_year = datetime.now().year

    prompt = f"""
    You are a smart assistant trained to select the best documents to answer a financial query about a company.

    <b>Available Document Types and Filename Formats:</b><br>
    <ul>
      <li><b>Annual Report:</b> Covers financial performance, revenue, net profit, balance sheet, etc.<br>
          Format example: <code>AnnualReport2023</code> (Year at the end).</li>
      <li><b>Concall Report:</b> Management commentary, earnings calls, outlook.<br>
          Format example: <code>Q4_2025</code> (Quarter + Year).</li>
      <li><b>Announcement:</b> Board decisions, dividends, buybacks, leadership changes.<br>
          Format example: <code>M&lt;month&gt;Y&lt;year&gt;D&lt;day&gt;</code>, e.g., <code>M6Y2025D3</code> means <b>3rd June 2025</b>.</li>
    </ul>

    <b>Instructions:</b><br>
    - Select the smallest, most focused set of documents, up to a maximum of <b>10</b>, to precisely answer the query.<br>

    - Prioritize only the most recent and highly relevant reports of the following types: <b>annual_report</b>, <b>concall</b>, and <b>announcement</b>.<br>

    - Use <b>annual reports</b> exclusively for detailed financial metrics such as performance, revenue, cash flow, capital expenditures (capex), and balance sheet items.<br>
 
    - Use <b>concall transcripts</b> strictly for management commentary, strategic initiatives, forward guidance, hiring updates, and business outlook.<br>
 
    - Use <b>announcements</b> solely for board decisions, dividends, acquisitions, leadership changes, buybacks, and other corporate actions.<br>
 
    - Focus on documents from the most recent fiscal years, specifically: <b>FY{current_year}</b>, <b>FY{current_year - 1}</b>, and <b>FY{current_year - 2}</b>.<br>
 
    - Ensure that documents referring to the same exact reporting date or period are not duplicated; only one document per unique date/period should be included.<br>
 
    - Avoid redundancy to maintain concise, accurate, and non-overlapping context for the query.

    <br>
    <b>Query:</b>  
    "{query}"

    <b>Available Documents:</b>
    {chr(10).join(f"- {label}" for label in list(set(doc_labels)))}

    Return a comma-separated list of up to 5 most relevant documents like, Concall - Q4_2025.pdf, Annual Report - AnnualReport2025.pdf etc . If no relevant match, return:
    "FALLBACK"

    Do not explain. Only return document labels or "FALLBACK".
    """

    # print(prompt)

    response = model.generate_content(prompt)
    result = response.text.strip()

    prompt_tokens = count_tokens(model, prompt)
    response_tokens = count_tokens(model, result)

    # If Gemini says "FALLBACK", apply fallback logic
    if "FALLBACK" in result.upper():
        # Step 1: Group by type and pick latest from each
        type_to_docs = defaultdict(list)
        for m in all_metadata:
            type_to_docs[m["type"]].append(m)

        fallback_docs = []
        for t in ["annual_report", "concall", "announcement"]:
            docs = sorted(
                type_to_docs.get(t, []),
                key=lambda x: x.get("source", ""),
                reverse=True,
            )
            if docs:
                fallback_docs.append(
                    f"{t.replace('_', ' ').title()} - {docs[0]['source']}"
                )
        return fallback_docs,{
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }
    

    reports = [doc.strip() for doc in result.split(",") if doc.strip()]
    reports = list(set(reports))
    return reports,{
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }
