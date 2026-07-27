import asyncio
import json
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


# Helpers
def load_tasks():
    if not os.path.exists("tasks.json"):
        return []
    with open("tasks.json", "r") as f:
        return json.load(f)


def save_tasks(tasks):
    with open("tasks.json", "w") as f:
        json.dump(tasks, f)


# Tools
@tool
def add_task(task: str) -> str:
    """
    Add new task.
    Args:
        task (str): The task to be added.
    Returns:
        str: Message that the task was added
    """

    tasks = load_tasks()

    tasks.append(task)
    save_tasks(tasks)

    return f"Added '{task}'"


@tool
def list_tasks() -> list[str]:
    """
    List all tasks.
    Returns:
        List of tasks
    """

    tasks = load_tasks()

    return tasks


@tool
def delete_task(index: int) -> str:
    """
    Delete a task by its number.
    Args:
        index (int): The index number of the task to be deleted (1-based index).
    Returns:
        str: Message confirming the deletion.
    """

    tasks = load_tasks()

    if index < 1 or index > len(tasks):
        return "Invalid task number."

    removed = tasks.pop(index - 1)
    save_tasks(tasks)

    return f"Deleted '{removed}'"

# Agent
agent = create_agent(
    model=f"openai:{os.getenv('OPENAI_MODEL')}",
    system_prompt="""
    You are a Todo Assistant. Use the available tools whenever needed.
    """,
    tools=[add_task, list_tasks, delete_task],
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"delete_task": True})],
    checkpointer=InMemorySaver(),
)


async def chat():
    config = {"configurable": {"thread_id": "1"}}

    while True:
        message = input("You: ")

        if message.lower() == "exit":
            break

        result = await agent.ainvoke(
            {"messages": [HumanMessage(message)]},
            config,
        )

        if "__interrupt__" in result:
            choice = input("Approve? (y/n): ")

            if choice.lower() == "y":
                result = await agent.ainvoke(
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    config,
                )
            else:
                result = await agent.ainvoke(
                    Command(
                        resume={
                            "decisions": [
                                {
                                    "type": "reject",
                                    "message": "User rejected the action.",
                                }
                            ]
                        }
                    ),
                    config,
                )

        print("AI:", result["messages"][-1].content)


asyncio.run(chat())
