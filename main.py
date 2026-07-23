from agents import (
    Agent,
    function_tool,
    SQLiteSession,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
)
import gradio as gr
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


# Tools
@function_tool
def check_order(order_id: str) -> str:
    """
    Use this tool to check the order status of any order.
    Args:
        order_id(str): ID of the order
    Return:
        Order Status
    """
    return "In Transit"


# Agnet
set_tracing_disabled(True)
agent = Agent(
    name="Business Agent",
    instructions="You are an AI Assistant for a business. Your job is to assist the users in their query related to the businees and order. Dont answer anything else.",
    tools=[check_order],
    model=OpenAIChatCompletionsModel(
        model=os.getenv("MODEL"),
        openai_client=AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ),
    ),
)

session = SQLiteSession("xyz")


async def handle_chat(message, _):
    res = await Runner.run(agent, message, session=session)
    return res.final_output


gr.ChatInterface(handle_chat).launch()
