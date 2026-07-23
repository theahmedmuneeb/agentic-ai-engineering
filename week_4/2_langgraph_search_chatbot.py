from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import Annotated
from typing_extensions import TypedDict
from langchain_exa import ExaSearchResults
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import gradio as gr
import resend

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")


# State
class State(TypedDict):
    messages: Annotated[list, add_messages]

    search: str = ""
    memory: str = ""


# LLM
llm = ChatOpenAI(
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("OPENAI_MODEL"),
)

# Tools
exa = ExaSearchResults()


@tool
def current_datetime() -> str:
    """Get the current date and time."""
    return str(datetime.now())


@tool
def send_email(message, subject):
    """
    Send an email to your owner. Just ask for the message and make subject yourself from the message
    Args:
        message (str): HTML Message of the email
        subject (str): Subject of the email
    """

    resend.Emails.send(
        {
            "from": os.getenv("RESEND_SENDER_EMAIL"),
            "to": os.getenv("RESEND_USER_EMAIL"),
            "subject": subject,
            "html": message,
        }
    )

    return "Email sent successfully!"


tools = [exa, current_datetime, send_email]

llm_with_tools = llm.bind_tools(tools)


# Nodes
def chatbot_node(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools)

# Build
builder = StateGraph(State)

builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")
builder.add_edge("chatbot", END)


# Gradio
def chat(message, _):
    config = {"configurable": {"thread_id": "conversation-1"}}

    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return result["messages"][-1].content


with SqliteSaver.from_conn_string("memory.db") as sql_memory:
    graph = builder.compile(checkpointer=sql_memory)

    gr.ChatInterface(chat).launch()


# import tempfile, os; f = tempfile.NamedTemporaryFile(delete=False, suffix=".png"); f.write(graph.get_graph().draw_mermaid_png()); f.close(); os.startfile(f.name)
