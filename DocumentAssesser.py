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

class RAG:
    def __init__(self, model_name = 'gemini-1.5-flash', embedding_model = "sentence-transformers/msmarco-distilbert-base-v4"):
        self.llm = ChatGoogleGenerativeAI(model=model_name,
                                          verbose=True,
                                          google_api_key=userdata.get('GOOGLE_API_KEY'),
                                          temperature = 0.3)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            show_progress=False
            )
        self.all_pages = []
        self.vectorDB = None
        self.retriever = None
        self.assesser = None

    def EmbedDocuments(self, paths):
        for path in paths:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            self.all_pages.extend(pages)
            print(f"Documents {path} loaded successfully!")

    def createVectorDB(self):
      if not self.all_pages:
          print("No documents loaded.")
          return
      if not self.embeddings:
          print("No embeddings provided.")
          return
      embeddings = self.embeddings.embed_documents([page.page_content for page in self.all_pages])
      if not embeddings:
          print("Failed to generate embeddings.")
          return
      self.vectorDB = FAISS.from_documents(self.all_pages, self.embeddings)
      print("Vector store created!")

    def createRetriever(self):
        self.retriever = VectorStoreRetriever(vectorstore=self.vectorDB)
        self.assesser = RetrievalQA.from_chain_type(llm=self.llm, retriever=self.retriever)
        print("Retriever created!")

    def setup(self, pdfs):
        self.EmbedDocuments(paths=pdfs)
        self.createVectorDB()
        self.createRetriever()

    def run(self, prompt):
      return self.assesser.run(prompt)

def get_assesser(llm_model: str, embedding_model: str, pdfs = list()):
	assesser = RAG(
		model_name=llm_model,
    	embedding_model=embedding_model
    	)
	assesser.setup(pdfs)
	return assesser

class customAgent:
  def __init__(self, llm, rag, max_iterations, goal, backstory):
    self.rag = rag
    self.llm = llm
    self.max_iterations = max_iterations
    self.goal = goal
    self.backstory = backstory
    self.previous_interactions = ""
    self.prompt = None
    self.task = None
    self.question = None
    self.answer = None
    self.final_output = None
  def ask_question(self):
    for i in range(self.max_iterations):
      self.prompt = f"""
        You are an agent and you will be assigned a task.
        Your backstory is: {self.backstory}
        Your goal is: {self.goal}
        Your previous interactions are: {self.previous_interactions}
        You get to ask the client only {self.max_iterations} questions so be very specific and concise.
      """
      self.task = "Please respond with a single question"
      self.prompt = self.prompt+self.task
      self.question = self.llm.invoke(self.prompt).content
      self.answer = self.rag.run(self.question)
      self.previous_interactions += f"Question: {self.question}\nAnswer: {self.answer}\n\n"
    self.prompt = f"""
        You are an agent and you will be assigned a task.
        Your backstory is: {self.backstory}
        Your goal is: {self.goal}
        Your previous interactions are: {self.previous_interactions}
        You get to ask the client only {self.max_iterations} questions so be very specific and concise.
      """
    self.task = "Please suggest your solution now based on your assessment of the client based on your interactions. Ask any relevant questions that can be forwarded to the client"
    self.prompt = self.prompt+self.task
    self.final_output = self.llm.invoke(self.prompt).content