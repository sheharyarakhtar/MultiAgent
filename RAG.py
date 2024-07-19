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