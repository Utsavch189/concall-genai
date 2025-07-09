from openai import OpenAI
import instructor
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

client = instructor.from_openai(OpenAI(api_key=os.getenv('OPENAI_API_KEy')))

gpt_mini = """
You are a financial analysis assistant. Analyze the full earnings conference call (concall) transcript provided below and extract summaries in the **exact format** described. Be structured, consistent, and follow the output format without adding anything extra.

Summarize these 4 sections:

1. quarterly_earnings_summary — approx 300 words  
2. new_projects_and_capex_planning — approx 300 words  
3. management_guidance — approx 300 words  
4. overall_summary — approx 150 words  

Each section must:
- Use bullet-point format (not numbered)
- Include sub-headings in <b></b> tags, like <b>Revenue:</b>
- Include values and key figures in <b></b> tags, like <b>₹200 crore</b> or <b>40% YoY</b>
- Do NOT include any section headings or keys like "quarterly_earnings_summary:" — just output the content

Maintain executive tone, financial insight, and structure — no markdown, no extra explanation.

Transcript:
{text}
"""


class ExtractedReport(BaseModel):
    company_name:str = "No available content found"
    quarterly_earnings_summary: str = Field(..., description="~300-word summary of Q4 FY25 earnings")
    new_projects_and_capex_planning: str = Field(..., description="~300-word detailed summary of new projects & capex planning")
    management_guidance: str = Field(..., description="~300-word detailed summary of management guidance and strategic outlook")
    overall_summary: str = Field(...,description="~150-word detailed summary of overall conclusion")

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

def prompt():
    try:
        chunks = split_pdf_into_chunks('blue star.pdf')
        text = "\n\n".join(chunk.page_content for chunk in chunks)
        # print("Input token usage --> ",client.count_tokens(contents=text).total_tokens)
        return f"""
            You are a financial analysis assistant. Analyze the full earnings conference call (concall) transcript provided below and extract summaries in the **exact format** described. Be structured, consistent, and follow the output format without adding anything extra.

Summarize these 4 sections:

1. quarterly_earnings_summary — approx 300 words  
2. new_projects_and_capex_planning — approx 300 words  
3. management_guidance — approx 300 words  
4. overall_summary — approx 150 words  

Each section must:
- Use bullet-point format (not numbered)
- Include sub-headings in <b></b> tags, like <b>Revenue:</b>
- Include values and key figures in <b></b> tags, like <b>₹200 crore</b> or <b>40% YoY</b>
- Do NOT include any section headings or keys like "quarterly_earnings_summary:" — just output the content

Maintain executive tone, financial insight, and structure — no markdown, no extra explanation.

Transcript:
{text}"""
    except Exception as e:
        print("Error : ",e)
        return ""

response = client.chat.completions.create(
    model = 'gpt-4.1-mini',
    messages=[
        {
            "role":"user",
            "content": prompt()
        }
    ],
    response_model=ExtractedReport
)

# print("Output Token : ",gemini_client.count_tokens(contents="\n".join(response.model_dump().values())).total_tokens)

with open('openai_4.1_mini_res1.json','w') as jsn:
    jsn.write(json.dumps(response.model_dump(),indent=2))
