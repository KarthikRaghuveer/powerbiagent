from fastapi import FastAPI
from pydantic import BaseModel
from langchain.utilities import PowerBIDataset
from powerbiclient.authentication import DeviceCodeLoginAuthentication
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import PowerBIToolkit, create_pbi_agent
import os

# Initialize FastAPI app
app = FastAPI()

# Define your dataset ID
DATASET_ID = 'c970b3b4-fdce-435b-aec7-5305ac01aa92'

# Initialize Power BI authentication
auth = DeviceCodeLoginAuthentication()
token = auth.get_access_token()

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Initialize PowerBI dataset
powerbi = PowerBIDataset(
    dataset_id=DATASET_ID,
    table_names=['Orders', 'People', 'Return'],
    token=token
)

# Initialize LLM
smart_llm = OpenAI(temperature=0, timeout=300)
toolkit = PowerBIToolkit(powerbi=powerbi,
                          llm=smart_llm, max_iterations=2,
                          output_token_limit=100)

# Create agent executor
agent_executor = create_pbi_agent(llm=smart_llm, toolkit=toolkit, verbose=True)

# Pydantic model for request body
class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Power BI Query API"}

@app.post("/query/")
async def run_query(request: QueryRequest):
    """
    Run a query against the Power BI dataset.
    """
    query = request.query
    result = agent_executor.run(query)
    return {"result": result}
