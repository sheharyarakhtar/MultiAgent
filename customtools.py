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
GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
from IPython.display import display, Markdown, Latex



@tool
def Assesser(question: str, assesser) -> str:
  """
  This tool allows you to query the provided documents.
  You can ask it relevant questions and it will return the answer.
  """
  prompt = f""""
  Context: You are the owner of the RFP documents you have access to.
  You will be asked questions regarding these documents.
  You answer these questions by assessing the documents you have access to.

  Question: {question}
  """
  return assesser.run(prompt)

@tool
def DuckDuckGoSearch(search_query: str):
    """Search the web for information on a given topic"""

    return DuckDuckGoSearchRun().run(search_query)

serper_tool = SerperDevTool()

import json

@tool
def SearchRFP(keywords: str, assesser) -> str:
    """
    Search the RFP documents with keywords

    keywords: str: Search keyword in the attached documents
    conducts a similarity search of document and results a dictionary of relevant content with page numbers
    """
    results = assesser.vectorDB.similarity_search(keywords)
    overall = {}
    content = ""
    for result in results:
      combine = {
          'Page Number':result.metadata['page'],
          'Content':result.page_content
      }
      overall[result.metadata['source']] = combine
      content = content + result.page_content + "\n\n\n"
    prettyjson = json.dumps(overall, indent=4)
    content = rephrase_text(content)
    return content

def rephrase_text(input_string, model):
  rephrased = model.invoke(
        f"""
        Context: You are a language model used to rephrase messy text.
        You will be given some messy text and your job is to format it and paraphrase it.
        Do not add anything yourself.
        Remove things that are not relevant i.e. any tables or idle letters. Just rephrase the content in a single or max 2 paragraphs

        Text: {input_string}
        """).content
  return rephrased

@tool
def write_to_file(input_string, file_path):
    """
    Write input_string to a text file at file_path.

    Parameters:
    - input_string (str): The string to write to the file.
    - file_path (str): The path where the file will be created or overwritten.

    Returns:
    - None
    """
    try:
        with open(file_path, 'w') as file:
            file.write(input_string)
        print(f"Successfully wrote to file: {file_path}")
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")


@tool
def pdf_search(query: str, pdf_path: str) -> str:
  """
  Pass in the query you wish you ask and the path to the PDF you wish to search the query in
  query: string
  pdf_path: string
  The function will return a string explaining what the pdf says about the query
  """

  return rag.run(query)