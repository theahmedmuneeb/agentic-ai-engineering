#!/usr/bin/env python
import os
import sys
import warnings

from datetime import datetime

from research_document_generator.crew import ResearchDocumentGenerator

from agents import Agent, Runner, set_tracing_disabled
from typing import Literal
from pydantic import BaseModel, Field
import asyncio
from agents.extensions.models.litellm_model import LitellmModel

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

set_tracing_disabled(True)


class Plan(BaseModel):
    topic: str = Field(description="The topic to be researched.")

    output: Literal["research", "pdf", "ppt"] = Field(
        description="The requested output format."
    )


def run():
    """
    Run the crew.
    """

    prompt = input("Prompt: ")

    planner = Agent(
        name="planner Agent",
        instructions="Analyze the user's request and extract the research topic and requested output format. The output format must be one of: 'research', 'pdf', or 'ppt'. Return 'pdf' if the user requests a PDF, 'ppt' if they request a PowerPoint or presentation, otherwise return 'research'.",
        output_type=Plan,
        model=LitellmModel(
            model=os.getenv("OPENAI_MODEL_NAME"),
        ),
    )

    plan_result = asyncio.run(Runner.run(planner, prompt))
    plan = plan_result.final_output

    print(f"Plan: {plan.model_dump_json()}")

    # Run Crew
    inputs = {
        "topic": plan.topic,
        "output": plan.output,
        "current_year": str(datetime.now().year),
    }

    try:
        generator = ResearchDocumentGenerator()

        research_task = generator.research_task()
        tasks = [research_task]

        if plan.output == "pdf":
            tasks.append(generator.pdf_task(context=[research_task]))
        elif plan.output == "ppt":
            tasks.append(generator.ppt_task(context=[research_task]))

        crew = generator.crew(tasks)

        crew.kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}
    try:
        ResearchDocumentGenerator().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        ResearchDocumentGenerator().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}

    try:
        ResearchDocumentGenerator().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception(
            "No trigger payload provided. Please provide JSON payload as argument."
        )

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": "",
    }

    try:
        result = ResearchDocumentGenerator().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
