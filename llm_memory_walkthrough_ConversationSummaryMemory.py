from langchain_core import chat_history
from langchain_groq import ChatGroq
from dotenv import load_dotenv; load_dotenv()
import os
from langchain_community.llms import Ollama
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm  = ChatGroq(model="qwen-qwq-32b")
llm_ollama = Ollama(model=os.getenv("OLLAMA_MODEL","mistral:latest"),base_url=os.getenv("OLLAMA_URL","http://localhost"))

summary_prompt_template = """<s><|user|>Summarize the conversations and update with the new lines.

Current summary:
{summary}

new lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""

summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template
)

# Define the type of memory we will use
memory = ConversationSummaryMemory(
    llm=llm_ollama,
    memory_key="summary",
    prompt=summary_prompt
)

# Chain the LLM, prompt, and memory together
llm_chain = LLMChain(
    prompt=summary_prompt,
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
            response = llm_chain.invoke({"new_lines":user_message})
            print(response['text'])



#other way to use memory
from dotenv import load_dotenv; load_dotenv()
import os

from langchain_ollama import OllamaLLM
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Setup LLM
llm_ollama = OllamaLLM(
    model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
    base_url=os.getenv("OLLAMA_URL", "http://localhost")
)

# Prompt Template
summary_prompt_template = PromptTemplate.from_template(
    """<s><|user|>Summarize the conversations and update with the new lines.

Current summary:
{summary}

new lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""
)

# Memory
memory = ConversationSummaryMemory(
    llm=llm_ollama,
    memory_key="chat_history",
    prompt=summary_prompt_template
)

# Combine prompt + llm into a runnable chain
chain = summary_prompt_template | llm_ollama

if __name__ == "__main__":
    while True:
        user_message = input("Enter Your Query: ")
        if user_message.lower() == "exit":
            print("Goodbye!")
            break
        else:
            # Store interaction in memory
            memory.save_context({"input": user_message}, {"output": ""})
            chat_history = memory.load_memory_variables({})["chat_history"]

            response = chain.invoke({"summary": chat_history, "new_lines": user_message})
            print(response)
