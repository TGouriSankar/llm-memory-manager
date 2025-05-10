from langchain_core import chat_history
from langchain_groq import ChatGroq
from dotenv import load_dotenv 
load_dotenv()
import os
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm  = ChatGroq(model="qwen-qwq-32b")

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


template = """<s><|user|>Current conversation:{chat_history}

{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt","chat_history"])


memory = ConversationBufferMemory(memory_key="chat_history")
llm_ollama = Ollama(model=os.getenv("OLLAMA_MODEL","mistral:latest"),base_url=os.getenv("OLLAMA_URL","http://localhost"))

llm_usage = LLMChain(
    prompt=prompt,
    llm=llm_ollama,
    memory=memory
    )


if __name__ == "__main__":
    while True:
        user_message = input("Enter Your Query:-  ")
        if user_message.lower() == "exit":
            print("Goodbye!")
            break
        else:
            response = llm_usage.invoke({"input_prompt":user_message})
            print(response['text'])
