import sqlite3
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser



def query_db(query):
    conn = sqlite3.connect('customer_database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    response = cursor.fetchall()
    print(response)
    result = pd.DataFrame(response)
    column_names = [description[0] for description in cursor.description]
    result.columns = column_names
    if 'InvoiceDate' in result.columns:
        result['InvoiceDate'] = pd.to_datetime(result['InvoiceDate'])
    conn.close()
    return result


def sql_agent_response(question, llm, db_name, table_name):
    sql_prompt = PromptTemplate(
        template = """
        System: You are an SQL Agent.
        You have access to a single database called {db_name}
        This database has only a single table called {table_name}.
        You will be prompted with a question from a user and your job is to write an SQL query that will retrieve the relevant data from the database.
        Assign proper aliases in your query.
        Additional info, you are using SQLite, so use the appropriate sql functions when required
        table schema is provided below:
            [(0, 'Invoice', 'TEXT', 0, None, 0),
             (1, 'StockCode', 'TEXT', 0, None, 0),
             (2, 'Description', 'TEXT', 0, None, 0),
             (3, 'Quantity', 'INTEGER', 0, None, 0),
             (4, 'InvoiceDate', 'TIMESTAMP', 0, None, 0),
             (5, 'Price', 'REAL', 0, None, 0),
             (6, 'Customer_ID', 'REAL', 0, None, 0),
             (7, 'Country', 'TEXT', 0, None, 0)]
        In your response, only return an SQL query in JSON format with nothing before or after it.
        ===========================================================================
        Question: {question}
        """,
        input_variables = ['db_name','table_name','columns','question']
    )
    sql_agent = sql_prompt | llm | JsonOutputParser()
    query = sql_agent.invoke({
        'db_name':db_name,
        'table_name':table_name,
        'question':question
    })['query']
    print(query)
    return query_db(query)