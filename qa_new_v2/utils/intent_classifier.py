import google.generativeai as genai
from datetime import datetime
from collections import defaultdict
import os

def count_tokens(model, text):
    return model.count_tokens(text).total_tokens

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_relevant_sources(query: str, all_metadata: list[dict]) -> list[str]:

    # def parse_filename(file_name):
    #     from calendar import month_abbr
    #     import re

    #     base_name = file_name.replace(".pdf", "")
    #     match_month = re.search(r"M(\d{1,2})Y(\d{4})D(\d{1,2})", base_name, re.IGNORECASE)
    #     if match_month:
    #         month_num, year, day = match_month.groups()
    #         try:
    #             return f"{month_abbr[int(month_num)]} {int(day)}, {year}"
    #         except IndexError:
    #             pass
    #     match_quarter = re.search(r"Q([1-4])[_\- ]?(\d{4})", base_name, re.IGNORECASE)
    #     if match_quarter:
    #         quarter, year = match_quarter.groups()
    #         return f"Q{quarter} FY {year}"
    #     return base_name

    doc_labels = [
        f"{m['type'].replace('_', ' ').title()} - {m['source']}"
        for m in all_metadata
    ]

    current_year = datetime.now().year

    prompt = f"""
    You are a smart assistant trained to select the best documents to answer a financial query about a company.

    <b>Instructions:</b>
    - Choose the smallest and most relevant set of up to 5 documents.
    - Prioritize the most recent relevant reports from the types: <b>annual_report</b>, <b>concall</b>, <b>announcement</b>.
    - Use annual reports for performance, revenue, cash flow, capex, etc.
    - Use concalls for management commentary, strategy, guidance, hiring, outlook.
    - Use announcements for board actions, dividends, acquisitions, leadership changes.
    - Prefer the most recent years, e.g., FY{current_year}, FY{current_year - 1}, FY{current_year - 2}.
    - Do NOT repeat documents of the same exact date.

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
        return fallback_docs
    
    prompt_tokens = count_tokens(model, prompt)
    response_tokens = count_tokens(model, result)

    # Otherwise, parse Gemini's returned document labels
    return [doc.strip() for doc in result.split(",") if doc.strip()],{
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }
