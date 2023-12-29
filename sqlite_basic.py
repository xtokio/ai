from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = OpenAI(openai_api_key="sk-your-openai-api-key",temperature=0, verbose=False)
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
db_chain.run("What employee title Michael has?")