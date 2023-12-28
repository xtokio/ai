from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

llm = Ollama(model="zephyr", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# Load data from website
print("Load data from website")
loader = WebBaseLoader('https://www.niddk.nih.gov/health-information/informacion-de-la-salud/diabetes/informacion-general/sintomas-causas')
data = loader.load()

# Split into chunks 
print("Split data into chunks")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
all_splits = text_splitter.split_documents(data)

print("Creates vector store")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
question = "Que significa la resistencia a la insulina?"

print(f"Ask question: > {question}")
qachain=RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
qachain({"query": question})