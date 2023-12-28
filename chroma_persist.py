from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma

# Load data from website
print("Load data from website")
loader = WebBaseLoader('https://en.wikipedia.org/wiki/Keanu_Reeves')
data = loader.load()

# Split into chunks 
print("Split data into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

print("Creates vector store")
vectordb = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(), persist_directory="basic_rag")
vectordb.persist()
vectordb = None