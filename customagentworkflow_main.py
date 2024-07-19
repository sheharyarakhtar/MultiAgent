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
#################### Custom Agent Workflow ###############
rag = RAG(
	model_name = 'gemini-1.5-flash',
	embedding_model_name = 'sentence-transformers/msmarco-distilbert-base-v4'
	)

agent = customAgent(
    llm = rag.llm,
    rag = rag,
    max_iterations = 15,
    goal = """
        Your goal is to find use cases related to data science, AI and machine learning.
        Find atleast 6 usecases.
        Do not be generic, be specific with your usecases.
        Give your output in the format <Usecase Name>: <Brief Detail>
        """,
    backstory = """
    You are a data science consultant.
    You have access to a tool which is a retrieval augmented generator
    This tool has read the documents sent over by the client and acts as the owner of the PDFs
    You can ask it relevant questions related to the document to determine where you can intervene.
    You cannot ask the same question more than once.
    """)
goals = agent.ask_question()
