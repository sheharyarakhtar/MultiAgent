from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, tool, PDFSearchTool
import pathlib
import textwrap
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from google.colab import userdata
from langchain.tools import DuckDuckGoSearchRun
genai.configure(api_key=GOOGLE_API_KEY)
from IPython.display import display, Markdown
from RAG import *
from DocumentAssesser import *
from customAgent import *
from customtools import *

############### CrewAI Agent Workflow ############

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', verbose=True,google_api_key=userdata.get('GOOGLE_API_KEY'))
rfp_assesser = Agent(
    role = 'RFP Assesser',
    goal="You are a Consultant in the {team_name} team. You need to ask relevant questions to the tool to assess if there are any use-cases related to you",
    backstory = (

        "You are a consultant in the {team_name} team"
        "Clients come to you with their request for proposals"
        "These documents contain their vision for their company, or problems they would like to solve"
        "You may ask relevant questions to the documents to assess its relevance to your field using the Assesser tool"
        "You dont present a solution for a problem that the client has not explicitly asked you to solve"
        "If there is no problem mentioned explicitly related to your field, you may use the Assesser tool to find which potential use-cases can be created"
        "You also have direct access to the RFP using the SearchRFP tool which results content directly from the document in a string format"
        "DO NOT ask for the entire RFP, only interact with it by asking questions and searching keywords"
        "DO NOT repeat questions, only ask one question once and only search a single keyword once."
        "Ask a maximum of 2-3 questions"
    ),
    verbose = True,
    memory = True,
    llm = assesser.llm,
    tools = [Assesser,  write_to_file],
    max_iterations = 5

)
summarise_usecases = Task(
    description=(
        "Ask questions relevant to you from the data using the tools assigned. You need to find all cases relevant to you as part of the {team_name} team"
        "The RFP does not answer all questions. Where the answers are unavailable, you make educated assumptions and move on."
        "You assess these documents to see where you, being part of the {team_name} team, can help solve some, if any, of these problems"
        ),
    expected_output = "A summary in markdown syntax detailing the use-cases that are relevant to your field, along with their page-numbers. Write these files using write_to_file tool",
    agent = agent,
    tools = [Assesser, write_to_file]
)

researcher = Agent(
    role = "Research Assistant",
    goal = "You will provide cutting edge solutions to the problems presented to you by RFP Assesser",
    backstory = (
        "Your coworker, RFP Assesser, has assessed the client's RFP and come up with potential places where {team_name} can be used"
        "You have access to a search tool which you can use to find relevant literature online"
        "You need to summarise each use-case into an achievable target"
        "Then you will search online to find relevant literature that will help you orchestrate each of these solutions"
    ),
    verbose = True,
    memory = True,
    llm = assesser.llm
)

specifying_task = Task(
    description = (
        "You are provided with a holistic view of where the client wants to make improvements."
        "You need to {num_solutions} solutions from these potential use-cases"
        "You need to be very specific about what problem is being solved and how its being solved"
    ),
    expected_output = "A title and a summary for each usecase presented, using the information gathered by your coworker RFP Assesser",
    agent = researcher
)

literature_review = Task(
    description = (
        "Now that you have specific tasks, you have been given the tools to search online for relevant literature"
        "Search whatever you think is relevant and support your use-cases with proper literature review"
    ),
    expected_output = "Title for each usecase with a paragraph and a URL to link each solution with relevant literature",
    agent = researcher,
    tools = [DuckDuckGoSearch]
)

crew = Crew(
    agents = [rfp_assesser, researcher],
    tasks = [summarise_usecases, specifying_task, literature_review],
    process = Process.sequential
)
result = crew.kickoff(inputs={'team_name': 'Data Science, AI and Machine Learning',
                     'team_tools': 'Data Science, AI and ML',
                              'num_solutions':'4'})