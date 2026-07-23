import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("OPENAI_MODEL"),
)

messages=[
    SystemMessage(content="Your name is Charles."),
]

prompt = input("Enter message: ")
messages.append(HumanMessage(prompt))

reply = llm.invoke(messages)

print(reply.content)
