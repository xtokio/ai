# How to run a basic chain
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    

llm = Ollama(model="zephyr", callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]), temperature=0.9)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(input_variables=["topic"], template="Give me 5 interesting facts about {topic}?")

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

# Run the chain only specifying the input variable.
chain.run("the moon")