from SQLAgent import *
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro', temprature = 0)
db_name = "customer_database"
table_name = "CRM"
question = "Whats our most expensive product and what is its price? Which country has it been bought the most?"

df = sql_agent_response(
    question = question, 
    llm = llm, 
    db_name = db_name, 
    table_name = table_name)
print(df)