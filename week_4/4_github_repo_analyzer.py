import base64
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import HumanMessage
import asyncio
from langchain_core.tools import tool
import requests
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


async def main():
    # Tools
    @tool
    def get_repository_info(owner: str, repo: str) -> dict:
        """
        Get Information for a GitHub repository.
        Args:
            owner (str): GitHub repository owner(Username/Organization)
            repo (str): GitHub repository name
        Returns:
            dict: A dictionary containing repository information.
        """
        try:
            response = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}",
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print("get_repository_info failed:", e)
            return f"Tool failed: {e}"

    @tool
    def get_repository_tree(owner: str, repo: str, branch: str) -> dict:
        """
        Get the file tree for a GitHub repository.
        Args:
            owner (str): GitHub repository owner(Username/Organization)
            repo (str): GitHub repository name
            branch (str): Branch name to get the file tree from(e.g. main, master)
        Returns:
            dict: A dictionary representing the file tree of the repository.
        """
        try:
            response = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}",
                params={
                    "recursive": 1,
                },
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print("get_repository_tree failed:", e)
            return f"Tool failed: {e}"

    @tool
    def read_file(owner: str, repo: str, path: str) -> str:
        """
        Read file from GitHub repository.
        Args:
            owner (str): GitHub repository owner(Username/Organization)
            repo (str): GitHub repository name
            path (str): Path to the file in the repository(e.g. src/main.py, README.md). Path should be alway a path of a file not a directory. If you speify a directory path then it will return an error and that will be counted as your failure.
        Returns:
            str: Content of the file
        """

        try:
            response = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            )
            response.raise_for_status()

            return base64.b64decode(response.json()["content"]).decode()

        except Exception as e:
            print("read_file failed:", e)
            return f"Tool failed: {e}"

    # Sandbox
    sandbox = os.path.abspath("sandbox")
    os.makedirs(sandbox, exist_ok=True)

    # Agents
    repo_analyzer = {
        "name": "repo_analyzer",
        "description": "Analyzes GitHub repositories by exploring relevant files and identifying architecture, technologies and key components.",
        "system_prompt": "You are a repository analyzer agent responsible for understanding codebases. Explore only the necessary files, identify the architecture, technologies and key components and return accurate structured findings.",
        "tools": [get_repository_tree, read_file],
    }

    report_writer = {
        "name": "report_writer",
        "description": "Writes technical reports based on repository analysis findings.",
        "system_prompt": "You are a technical report writer responsible for presenting repository analysis Transform the analyzer's findings into a concise, well-structured Markdown report without adding unsupported information.",
    }

    coordinator = create_deep_agent(
        name="coordinator",
        model=ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_MODEL"),
        ),
        system_prompt="You are a coordinator agent responsible for orchestrating repository analysis. Verify the repository exists using the get_repository_info tool, pass the repository context including the owner, repository name, default branch and other relevant metadata to the repo_analyzer. Delegate report generation to the report_writer and return the final response without analyzing the repository yourself. Only delegate if you have enough context and there is no previous error (e.g. If the repository does not exist return an error message without delegating ).",
        subagents=[repo_analyzer, report_writer],
        tools=[get_repository_info],
        checkpointer=MemorySaver(),
        backend=FilesystemBackend(root_dir=sandbox, virtual_mode=True),
    )

    # Logic
    print("Type 'exit' to quit. \n")
    config = {"configurable": {"thread_id": "thread-1"}}

    while True:
        prompt = input("You: ")
        if prompt == "exit":
            print("Exiting...")
            break

        res = await coordinator.ainvoke(
            {"messages": [HumanMessage(prompt)]}, config=config
        )

        print("AI:", res["messages"][-1].content)


asyncio.run(main())
