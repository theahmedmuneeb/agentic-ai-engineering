import os
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerHTTP
from crewai_tools import ExaSearchTool
from .tools.execute_python import execute_python_code
import yaml
from pathlib import Path


# @CrewBase
class ResearchDocumentGenerator:
    """ResearchDocumentGenerator crew"""

    def __init__(self):
        # Load Agents and Tasks Config
        config_dir = Path(__file__).parent / "config"

        with open(config_dir / "agents.yaml", "r") as f:
            self.agents_config = yaml.safe_load(f)

        with open(config_dir / "tasks.yaml", "r") as f:
            self.tasks_config = yaml.safe_load(f)

        # Context7 MCP
        context7 = MCPServerHTTP(
            url="https://mcp.context7.com/mcp",
            headers={
                "CONTEXT7_API_KEY": os.getenv("CONTEXT7_API_KEY"),
            },
            streamable=True,
        )

        # Agents

        self.manager_agent = Agent(
            config=self.agents_config["manager"],
            verbose=True,
        )

        self.researcher_agent = Agent(
            config=self.agents_config["researcher"],
            verbose=True,
            tools=[ExaSearchTool()],
        )

        self.pdf_agent = Agent(
            config=self.agents_config["pdf_agent"],
            verbose=True,
            tools=[execute_python_code],
            mcps=[context7],
        )

        self.ppt_agent = Agent(
            config=self.agents_config["ppt_agent"],
            verbose=True,
            tools=[execute_python_code],
            mcps=[context7],
        )

    # Agents
    # def manager(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["manager"],
    #         verbose=True,
    #     )

    # # @agent
    # def researcher(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["researcher"],
    #         verbose=True,
    #         tools=[ExaSearchTool()],
    #     )

    # # @agent
    # def pdf_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["pdf_agent"],
    #         verbose=True,
    #         tools=[execute_python_code],
    #     )

    # # @agent
    # def ppt_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["ppt_agent"],
    #         verbose=True,
    #         tools=[execute_python_code],
    #     )

    # Tasks
    # @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.researcher_agent,
        )

    # @task
    def pdf_task(self, context: list[Task]) -> Task:
        return Task(
            config=self.tasks_config["pdf_task"],
            agent=self.pdf_agent,
            context=context,
        )

    # @task
    def ppt_task(self, context: list[Task]) -> Task:
        return Task(
            config=self.tasks_config["ppt_task"],
            agent=self.ppt_agent,
            context=context,
        )

    # Crew
    # @crew
    def crew(self, tasks: list[Task]) -> Crew:
        """Creates the ResearchDocumentGenerator crew"""
        return Crew(
            agents=[
                self.researcher_agent,
                self.pdf_agent,
                self.ppt_agent,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            # manager_agent=self.manager(),
        )
