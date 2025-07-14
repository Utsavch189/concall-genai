import google.generativeai as genai

def count_tokens(model, text):
    return model.count_tokens(text).total_tokens

def get_doc_types(query: str) -> list[str]:
    prompt = f"""
    You are a smart multi-label document classifier for financial queries.
    Your task is to determine which types of company documents are relevant to answer the user's question.

    Choose one or more from the following (return as comma-separated, no explanation, no extra text):
    - annual_report - – Use this if the query is about financial performance, revenue, net profit, cash flow, balance sheet, capital expenditure, ESG, business model, or any structured yearly reporting.
    - concall - – Use this if the query involves management discussion, strategy, future guidance, sentiment, plans, margins, investments, hiring, or commentary from earnings calls.
    - announcement - – Use this if the query involves company updates like dividends, board meetings, acquisitions, new projects, leadership changes, expansion, press releases, or compliance filings.

    It's not necessary always return multiple doc type.
    Based on your smartness you will return only relevant ones.
    It might be only annual or annual and concall or all the three etc.
    ---

    **Query:**  
    "{query}"

    ---

    Return only relevant document types in lowercase and comma-separated.
    """

    model = genai.GenerativeModel("models/gemini-2.5-flash")

    response = model.generate_content(prompt)
    result = response.text.strip()

    prompt_tokens = count_tokens(model, prompt)
    response_tokens = count_tokens(model, result)

    return [d.strip() for d in result.split(",")],{
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }
