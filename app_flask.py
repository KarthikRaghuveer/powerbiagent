from flask import Flask, request, jsonify
from langchain.chat_models import AzureChatOpenAI
import openai
from langchain.utilities import PowerBIDataset
from langchain.llms import OpenAI
from powerbiclient.authentication import DeviceCodeLoginAuthentication
from langchain.agents.agent_toolkits import PowerBIToolkit, create_pbi_agent
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize Power BI authentication
auth = DeviceCodeLoginAuthentication()
token = auth.get_access_token()

# Define your dataset ID
DATASET_ID = 'c970b3b4-fdce-435b-aec7-5305ac01aa92'

# Initialize PowerBI dataset
powerbi = PowerBIDataset(
    dataset_id=DATASET_ID,
    table_names=['Orders', 'People', 'Return'],
    token=token
)

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'

# Initialize LLM
smart_llm = OpenAI(temperature=0, timeout=300)
toolkit = PowerBIToolkit(powerbi=powerbi,
                          llm=smart_llm, max_iterations=2,
                          output_token_limit=100)

# Create agent executor
agent_executor = create_pbi_agent(llm=smart_llm, toolkit=toolkit, verbose=True)

@app.route('/')
def index():
    return 'Welcome to the Power BI Query API'

@app.route('/query/', methods=['POST'])
def run_query():
    if request.method == 'POST':
        query = request.json.get('query', '')
        if query:
            result = agent_executor.run(query)
            return jsonify({"result": result})
        else:
            return jsonify({"error": "Query is required"}), 400
    else:
        return jsonify({"error": "Method Not Allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)
