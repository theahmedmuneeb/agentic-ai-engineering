from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()


async def main():
    # Tools
    browser_mcp = MultiServerMCPClient(
        {
            "playwright": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@playwright/mcp@latest", "--isolated"],
            }
        }
    )
    browser_tools = await browser_mcp.get_tools()

    # Agent
    agent = create_agent(
        model=f"openai:{os.getenv('OPENAI_MODEL')}",
        system_prompt="""
        You are a helpful AI assistant. You answers consiely and short responses.
        
        Tools:
            Browser Tools:
            You have access to browser tools that can be used to perform web related operations. You should use the tools to perform the tasks that the user asks you to do. If in the search results you find the answer and its the thing what user asked then return the answer. Dont do too much searches or web operations. e.g. User asked for something and you searched and found the answer then dont do more operation or visiting extra webpages. But make sure that the result is accurate. Browser is not only for Searching but also any other things that possible and user says.
        """,
        tools=[*browser_tools],
        checkpointer=MemorySaver(),
    )

    #
    print("Type 'exit' to quit. \n")
    config = {"configurable": {"thread_id": "thread-1"}}

    while True:
        prompt = input("You: ")
        if prompt == "exit":
            print("Exiting...")
            break

        res = await agent.ainvoke({"messages": [HumanMessage(prompt)]}, config=config)

        print("AI:", res["messages"][-1].content)


asyncio.run(main())
