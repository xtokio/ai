from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

llm = Ollama(model="zephyr", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

vectordb = Chroma(persist_directory="basic_rag", embedding_function=GPT4AllEmbeddings())

qachain=RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever())
question = "When did Reeves reunited with Winona Ryder for the comedy Destination Wedding?"
print(question)
qachain({"query": question})