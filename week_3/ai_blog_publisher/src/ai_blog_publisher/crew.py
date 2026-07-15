from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
import os
from crewai_tools import ExaSearchTool
from crewai.tools import tool
from urllib.parse import quote
import requests


# Tools
@tool("Generate a featured image")
def generate_featured_image(prompt: str) -> str:
    """
    Generate the featured image for blog post.
    Args:
        prompt: The prompt for the image generation.
    Returns:
        The URL of the generated image.
    """
    return f"https://image.pollinations.ai/prompt/{quote(prompt)}?model=flux&width=1200&height=630"


@tool("Publish the article")
async def publish_article(
    title: str,
    body_markdown: str,
    description: str,
    tags: list[str],
    cover_image: str,
) -> str:
    """
    Publish an article
    Args:
        title: The title of the article.
        body_markdown: The content of the article in Markdown format.
        description: A short summary of the article used for previews and SEO meta description.
        tags: A list of tags (up to 4 tags) to categorize the post and improve discoverability.
        cover_image: Absolute URL of the cover image for the article.
    Returns:
        The URL of the published article.
    """

    res = requests.post(
        "https://dev.to/api/articles",
        headers={
            "api-key": os.environ["DEVTO_API_KEY"],
            "Content-Type": "application/json",
        },
        json={
            "article": {
                "title": title,
                "body_markdown": body_markdown,
                "description": description,
                "tags": tags,
                "main_image": cover_image,
                "published": True,
            }
        },
    )

    res.raise_for_status()

    return res.json()["url"]


@CrewBase
class AiBlogPublisher:
    """AI Blog Publisher crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            tools=[ExaSearchTool()],
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config["writer"],  # type: ignore[index]
            verbose=True,
        )

    @agent
    def publisher(self) -> Agent:
        return Agent(
            config=self.agents_config["publisher"],  # type: ignore[index]
            verbose=True,
            tools=[generate_featured_image, publish_article],
        )

    # tasks
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["writing_task"],
        )

    @task
    def publishing_task(self) -> Task:
        return Task(
            config=self.tasks_config["publishing_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiBlogPublisher crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
