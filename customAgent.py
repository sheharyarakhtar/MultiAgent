from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from google.colab import userdata
genai.configure(api_key=userdata.get('GOOGLE_API_KEY'))
os.environ['GOOGLE_API_KEY']=userdata.get('GOOGLE_API_KEY')

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