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
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
from IPython.display import display, Markdown
os.environ['GOOGLE_API_KEY']=userdata.get('GOOGLE_API_KEY')
from customAgent import *
from RAG import *

def get_assesser(llm_model: str, embedding_model: str, pdfs = list()):
	assesser = RAG(
		model_name=llm_model,
    	embedding_model=embedding_model
    	)
	assesser.setup(pdfs)
	return assesser

