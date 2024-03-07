import streamlit as st
from langchain.chat_models import AzureChatOpenAI
import openai
from langchain.utilities import PowerBIDataset
from langchain.llms import OpenAI
from powerbiclient.authentication import DeviceCodeLoginAuthentication
from langchain.agents.agent_toolkits import PowerBIToolkit, create_pbi_agent
import os

# Initialize Power BI authentication
auth = DeviceCodeLoginAuthentication()
token = auth.get_access_token()
# +token='eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IlhSdmtvOFA3QTNVYVdTblU3Yk05blQwTWpoQSIsImtpZCI6IlhSdmtvOFA3QTNVYVdTblU3Yk05blQwTWpoQSJ9.eyJhdWQiOiJodHRwczovL2FuYWx5c2lzLndpbmRvd3MubmV0L3Bvd2VyYmkvYXBpIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvN2MzNzk0YjgtMmFhMi00Zjc3LWJlMzMtNDkxMGYwZTVmYjE4LyIsImlhdCI6MTcwOTgxMjIwNiwibmJmIjoxNzA5ODEyMjA2LCJleHAiOjE3MDk4MTYzODIsImFjY3QiOjAsImFjciI6IjEiLCJhaW8iOiJBVFFBeS84V0FBQUFURXdkQnVkeW9PZGtSMjBYS1RRT3RrWWJTN1Yyc3NsNXlKTE9Xc3Y3ZW9Tc3hTZUZOVjBrK1ZZdnNhUFVPcVVoIiwiYW1yIjpbInB3ZCJdLCJhcHBpZCI6IjFhZWEzZjk3LWVkYzYtNDQ1My1hNTliLWI4OGIwYjgwMzcxMSIsImFwcGlkYWNyIjoiMCIsImZhbWlseV9uYW1lIjoiUiIsImdpdmVuX25hbWUiOiJLYXJ0aGlrIiwiaXBhZGRyIjoiMjQwOTo0MGYyOjIwODg6MjU2MTplNDU2OjcxMGM6YWI0ZTplOTA3IiwibmFtZSI6IkthcnRoaWsgUiIsIm9pZCI6IjQ5Yzk5NzcxLWRhOTMtNDJiNS1hM2IxLTUwNWYzYjA5NjdmYyIsInB1aWQiOiIxMDAzMjAwMzVBNTVFREZGIiwicmgiOiIwLkFTc0F1SlEzZktJcWQwLS1NMGtROE9YN0dBa0FBQUFBQUFBQXdBQUFBQUFBQUFEQ0FOQS4iLCJzY3AiOiJDb250ZW50LkNyZWF0ZSBEYXRhc2V0LlJlYWRXcml0ZS5BbGwgUmVwb3J0LlJlYWRXcml0ZS5BbGwgV29ya3NwYWNlLlJlYWQuQWxsIiwic3ViIjoiYmJ6MmtxUEJOc2RXRFctbkIzZ1VMWUtveVlSTUhQZkg0TzF1LW01MHd0ayIsInRpZCI6IjdjMzc5NGI4LTJhYTItNGY3Ny1iZTMzLTQ5MTBmMGU1ZmIxOCIsInVuaXF1ZV9uYW1lIjoia2FydGhpay5yQGthcnZlbnR1bS5jb20iLCJ1cG4iOiJrYXJ0aGlrLnJAa2FydmVudHVtLmNvbSIsInV0aSI6IjhBelFYLWQ3X0V5U29WNXRqUlFqQUEiLCJ2ZXIiOiIxLjAiLCJ3aWRzIjpbImI3OWZiZjRkLTNlZjktNDY4OS04MTQzLTc2YjE5NGU4NTUwOSJdfQ.bDKh-N4-0X0HCzyi4zkOsUpzjPZbK5B-9QnypcEygayuvyA7sVRXVCyNgWjNRy-KGMdOijBLjjykFyAJdnNAcUqZkzL1VURB73yl20YXsQRwO45RynSmELZnMkUz9gRv7OOthb_oyMyT6EQsQ9EoDquFC4pdY6EkjgUQ50xApo6PP6VGFOL8SQLbIhhvSyCA5AQzBFMKU-prsrmGv9TngBRhNLWTl3IdCww-ldUb7F6eOilu3bDc1Co_oEwgwtxpkhjKOMdFlwpoBqeTl-wMoMEBi0P0TpXx4sr6yYanP3Gv0HHvv29m_YBG3nZSWpmSaK2yRLdAvCsvq2JdlYIgrQ'

# Define your dataset ID
DATASET_ID = 'c970b3b4-fdce-435b-aec7-5305ac01aa92'

# Initialize PowerBI dataset
powerbi = PowerBIDataset(
    dataset_id=DATASET_ID,
    table_names=['Orders', 'People', 'Return'],
    token=token
)

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = ''

# Initialize LLM
smart_llm = OpenAI(temperature=0, timeout=300)
toolkit = PowerBIToolkit(powerbi=powerbi,
                          llm=smart_llm, max_iterations=10,
                          output_token_limit=100)

# Create agent executor
agent_executor = create_pbi_agent(llm=smart_llm, toolkit=toolkit, verbose=True)

# Streamlit app
st.title('Power BI Query API')

query = st.text_input('Enter your query:')
if st.button('Run Query'):
    if query:
        result = agent_executor.run(query)
        st.write('Result:', result)
    else:
        st.error('Please enter a query')
