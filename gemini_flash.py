import instructor
from dotenv import load_dotenv
import google.generativeai as genai
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from pydantic import BaseModel,Field,field_validator

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

gemini_client = genai.GenerativeModel(
        model_name='models/gemini-2.5-flash'
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
        chunks = split_pdf_into_chunks('tips_re.pdf')
        text = "\n\n".join(chunk.page_content for chunk in chunks)
        print("Input token usage --> ",gemini_client.count_tokens(contents=text).total_tokens)
        return f"""
            You are a financial analyst assistant. Analyze the entire text of the company’s 
            quarterly earnings conference call (concall) transcript provided below. 
            From this transcript, extract and summarize the following three key areas. 
            Your summaries must be highly insightful, context-rich.
            Focus on clarity, financial relevance, and executive tone.

            **company_name**  
                > The official name of the company holding the earnings call. 
                This is usually mentioned at the beginning of the transcript or in opening remarks.
                
            ### Output format [Example] :
              1. Quarterly Earnings Summary (**quarterly_earnings_summary**):

                > Need To Follow Strictly:
                 1. Include values and important figures and key parts in <b></b> tags, like <b>₹200 crore</b> or <b>40% YoY</b> or <b>146 stores</b> or <b>2,100+ service</b> or <b>15%</b> or <b>1.53 million</b> .
                 2. Use bullet-point format (not numbered)
                 3. Each sub-headings must be '\n' seperated.
                 4. Include sub-headings in <b></b> tags, like <b>Revenue:</b> 

                > Try To Include below sub headings with insightful content and each sub heading must have approx 150-200 words and '\n' seperated each of sub-headings:
                 Revenue: ₹788 crore, down 1.2% YoY due to muted consumer demand.
                 Gross Margin: ₹455 crore; margin contracted by 230 bps YoY due to product mix (higher franchise and e‑commerce) and value proposition pricing.
                 EBITDA Margin: 25.5% (adjusted 23.5% after accounting changes); declined 14 bps YoY.
                 PAT: ₹46 crore; declined by 215 bps YoY.
                 Inventory: Reduced by 16% YoY; aged inventory reduced by 30–35%.
                 Same‑store metrics (ZBM initiative):
                 Inventory lines down 40%, size availability up 300 bps.
                 Retrieval time cut to 45 seconds (vs. 1.5 minutes earlier).
                 Volume growth from ZBM stores in mid‑single digits.

              2. New Projects & Capex Planning (**new_projects_and_capex_planning**):
                
                > Need To Follow Strictly:
                 1. Include values and important figures and key parts in <b></b> tags, like <b>₹200 crore</b> or <b>40% YoY</b> or <b>146 stores</b> or <b>2,100+ service</b> or <b>15%</b> or <b>1.53 million</b>  .
                 2. Use bullet-point format (not numbered)
                 3. Each sub-headings must be '\n' seperated.
                 4. Include sub-headings in <b></b> tags, like <b>Strategy:</b>

                > Try To Include below sub headings with insightful content and each sub heading must have approx 150-200 words and '\n' seperated each of sub-headings:
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

              3. Management Guidance (**management_guidance**) :

               > Need To Follow Strictly:
                 1. Include values and important figures and key parts in <b></b> tags, like <b>₹200 crore</b> or <b>40% YoY</b> or <b>146 stores</b> or <b>2,100+ service</b> or <b>15%</b> or <b>1.53 million</b> .
                 2. Use bullet-point format (not numbered)
                 3. Each sub-headings must be '\n' seperated.
                 4. Include sub-headings in <b></b> tags, like <b>Demand Outlook:</b>

               > Try To Include below sub headings with insightful content and each sub heading must have approx 150-200 words and '\n' seperated each of sub-headings:
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

            4. Overall Summary (**overall_summary**) [Total ~150 words and precise] :
                > Try To Include below with insightful content and must have approx 150-200 words and Include values and important figures and key parts in <b></b> tags, like <b>₹200 crore</b> or <b>40% YoY</b> or <b>146 stores</b> or <b>2,100+ service</b> or <b>15%</b> or <b>1.53 million</b> :
                    For example of Bata Company,
                    Bata India’s latest quarterly performance reflects resilience amid a subdued consumer demand environment. 
                    While revenue declined slightly by 1.2% YoY, volume growth and effective inventory management signal operational strength. 
                    Margin pressures due to strategic channel shifts and value pricing are being tackled through structural cost resets and 
                    backend efficiencies. The company’s sharp focus on innovation, store modernization through initiatives like ZBM, 
                    and aggressive expansion plans—especially in underpenetrated markets—underscore a growth-oriented outlook. With its 
                    ‘Customer First’ transformation and emerging export potential, Bata is positioning itself for long-term value creation through 
                    a balanced play of affordability, premiumization, and operational agility.  

            Maintain executive tone, financial insight, and structure — no markdown, no extra explanation.

            Transcript:
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
# raw_text = response.choices[0].message.content
print(dir(response))
print("Output Token : ",gemini_client.count_tokens(contents="\n".join(response.model_dump().values())).total_tokens)

with open('flash_res3.json','w') as jsn:
    jsn.write(json.dumps(response.model_dump(),indent=2))
