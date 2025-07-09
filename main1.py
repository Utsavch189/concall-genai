import instructor
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from pydantic import BaseModel,Field,field_validator


"""
            You are a financial analyst assistant. Analyze the entire text of the company’s quarterly earnings conference call (concall) transcript provided below. 
            From this transcript, extract and summarize the following three key areas. 
            Your summaries must be highly insightful, context-rich, and approximately 300 words each. 
            Focus on clarity, financial relevance, and executive tone.

            1. **company_name**  
                > The official name of the company holding the earnings call. 
                This is usually mentioned at the beginning of the transcript or in opening remarks.

            2. Quarterly Earnings Summary (**quarterly_earnings_summary**): 
                > Provide a comprehensive summary of this quarter’s financial performance. 
                Include key metrics discussed such as revenue, EBITDA, PAT, margins, YoY or QoQ comparisons, 
                and any major contributors or detractors to performance. 
                Reflect the management's tone (cautious, optimistic, etc.) and note any anomalies, trends, or unexpected results.

            3. New Projects and Capex Planning (**new_projects_and_capex_planning**):
                > Identify and summarize all references to upcoming or ongoing projects, capital expenditure plans, 
                investment initiatives, capacity expansions, R&D efforts, and geographic expansions. Mention financial figures (if disclosed),
                timelines, strategic rationale, and management’s stated objectives for these initiatives.

            4. Management Guidance (**management_guidance**):
                > Extract the management’s forward-looking commentary, including guidance on revenue, margin expectations, business environment,
                regulatory or macroeconomic risks, and demand outlook. Include any segment-wise commentary or strategic direction 
                for the coming quarters.
            
            ### Output format:
            1. Quarterly Earnings Summary (**quarterly_earnings_summary**):
            [~300-word detailed summary]

            2. New Projects & Capex Planning (**new_projects_and_capex_planning**):
            [~300-word detailed summary]

            3. Management Guidance (**management_guidance**):
            [~300-word detailed summary]
            
            ### Transcript Content:
            {text}"""

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

gemini_client = genai.GenerativeModel(
        model_name='models/gemini-2.5-pro'
    )

class ExtractedReport(BaseModel):
    company_name:str = "No available content found"
    quarterly_earnings_summary: str = Field(..., description="~300-word summary of Q4 FY25 earnings")
    new_projects_and_capex_planning: str = Field(..., description="~300-word detailed summary of new projects & capex planning")
    management_guidance: str = Field(..., description="~300-word detailed summary of management guidance and strategic outlook")
    overall_summary: str = Field(...,description="~150-word detailed summary of overall conclusion")

client = instructor.from_gemini(
    gemini_client,
    mode=instructor.Mode.GEMINI_JSON
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

def prompt():
    try:
        chunks = split_pdf_into_chunks('blue star.pdf')
        text = "\n\n".join(chunk.page_content for chunk in chunks)
        print("Input token usage --> ",gemini_client.count_tokens(contents=text).total_tokens)
        return f"""
            You are a financial analyst assistant. Analyze the entire text of the company’s quarterly earnings conference call (concall) transcript provided below. 
            From this transcript, extract and summarize the following three key areas. 
            Your summaries must be highly insightful, context-rich, and approximately 300 words each. 
            Focus on clarity, financial relevance, and executive tone.

            **company_name**  
                > The official name of the company holding the earnings call. 
                This is usually mentioned at the beginning of the transcript or in opening remarks.
            
            Each field’s content must be ~300 words and formatted in plain-text replicating the style below:
                
            ### Output format [Example]:
            1. Quarterly Earnings Summary (**quarterly_earnings_summary**) [Total ~300 words]:
            Revenue: ₹788 crore, down 1.2% YoY due to muted consumer demand.
            Gross Margin: ₹455 crore; margin contracted by 230 bps YoY due to product mix (higher franchise and e‑commerce) and value proposition pricing.
            EBITDA Margin: 25.5% (adjusted 23.5% after accounting changes); declined 14 bps YoY.
            PAT: ₹46 crore; declined by 215 bps YoY.
            Inventory: Reduced by 16% YoY; aged inventory reduced by 30–35%.
            Same‑store metrics (ZBM initiative):
            Inventory lines down 40%, size availability up 300 bps.
            Retrieval time cut to 45 seconds (vs. 1.5 minutes earlier).
            Volume growth from ZBM stores in mid‑single digits.

            2. New Projects & Capex Planning (**new_projects_and_capex_planning**) [Total ~300 words]:
            Zero Base Merchandising (ZBM):
            Expanded to 146 stores (targeting 300 by December 2025).
            Aims to declutter stores, improve consumer experience, and optimize inventory.
            Portfolio Innovation:
            Floatz: Strong growth (>40% YoY), aiming ₹200 crore revenue in FY26.
            Power: Expanded with Move+, EasySlide, and premium “Stamina+” lines.
            Hush Puppies: Premium positioning via campaigns; expanding store footprint.
            Customer First Program:
            A major transformation project focusing on data-led decision making, agility, and consumer centricity.
            Capex:
            Largest-ever backend investment at Batanagar: new PUDIP machine (advanced manufacturing).
            Strategy: automate and own high-tech manufacturing; outsource labor-intensive parts.

            3. Management Guidance (**management_guidance**) [Total ~300 words]:
            Store Expansion:
            FY25 saw 100 new stores; FY26 will see higher addition.
            Maintain 80:20 mix between franchise and company-owned stores.
            Demand Outlook:
            Cautiously optimistic despite muted environment.
            Focus on volume-driven growth over pricing-led growth.
            Pricing to be adjusted via cost optimization and product value.
            Channel Mix (FY25):
            COCO: ~70%, Franchise: 7.5%, E-commerce: 10%, Distribution (IND): ~12–13%.
            Gross Margin Strategy:
            Will stabilize over time with improved cost structures and value positioning.
            Premiumization (via Power & Hush Puppies) and mass value segments to run in parallel.
            Inventory & Working Capital:
            Further optimization planned.
            Aged inventory already at industry best-in-class (~2–3% of total).
            Export Opportunity:
            Post BIS norms, Bata India is 100% localized and aims to export to Bata Global in future.

            4. Overall Summary (**overall_summary**) [Total ~150 words and precise]:
            For example of Bata Company,
            Bata India’s latest quarterly performance reflects resilience amid a subdued consumer demand environment. 
            While revenue declined slightly by 1.2% YoY, volume growth and effective inventory management signal operational strength. 
            Margin pressures due to strategic channel shifts and value pricing are being tackled through structural cost resets and 
            backend efficiencies. The company’s sharp focus on innovation, store modernization through initiatives like ZBM, 
            and aggressive expansion plans—especially in underpenetrated markets—underscore a growth-oriented outlook. With its 
            ‘Customer First’ transformation and emerging export potential, Bata is positioning itself for long-term value creation through 
            a balanced play of affordability, premiumization, and operational agility.

            Now generate all three sections using exactly that tone and bullet‑like style (not numbered, no headings), separated by blank lines within each field. Do **not** include any additional keys, commentary, or header text.
            Try to add all sub parameters that are mentioned with '\n' seperated under those 3 main categories.
            Try to add <b></b> around the sub parameters like <b>Export Opportunity:</b>.
            Try to add <b></b> around the important places like growth numbers or any value or important things like <b>₹200 crore</b>.
            
            ### Transcript Content:
            {text}"""
    except Exception as e:
        print("Error : ",e)
        return ""

response = client.messages.create(
    messages=[
        {
            "role":"user",
            "content": prompt()
        }
    ],
    response_model=ExtractedReport
)

print("Output Token : ",gemini_client.count_tokens(contents="\n".join(response.model_dump().values())).total_tokens)

with open('res5.json','w') as jsn:
    jsn.write(json.dumps(response.model_dump(),indent=2))
