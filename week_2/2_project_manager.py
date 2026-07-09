from nicegui import ui
from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
)
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()


@ui.page("/")
def index():
    openai_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    model = OpenAIChatCompletionsModel(
        model=os.getenv("OPENAI_MODEL"), openai_client=openai_client
    )
    set_tracing_disabled(True)

    # Agents
    requirements_agent = Agent(
        name="Requirements Agent",
        model=model,
        instructions="You are a business expert. you have to generate things like Functional requirements, Non-functional requirements, User roles and core features etc from a business idea given to you. you have to return a final markdown with all the things",
    )

    architecture_agent = Agent(
        name="Architecture Agent",
        model=model,
        instructions="You are a software architect. your responsibility is to generate an overall architecture, Tech stach, file/folder structure and components. Return the fnal markdown.",
    )

    database_agent = Agent(
        name="Database Agent",
        model=model,
        instructions="You are a database engineer. Your responsibility is to generate database design, schema, tables, relatinships, primary and foreign keys etc.",
    )

    api_agent = Agent(
        name="API Agent",
        model=model,
        instructions="You are a backend engineer. Your responsibility is to generate API design, endpoints, request and response format, authentication and authorization etc. You have to follow best practices.",
    )

    reviewer_agent = Agent(
        name="Reviewer Agent",
        model=model,
        instructions="You are a senior software reviewer. Your job is to review all documents and info provided. check for missing requirements, inconsistencies and better choices and produce one final polished software design document.",
    )

    frontend_agent = Agent(
        name="Frontend Agent",
        model=model,
        instructions="You are a frontend engineer. Your responsibility is to generate frontend design, components, pages, routing, state management and UI/UX design. You have to follow best practices. You have to return a final markdown with all the things. No code is required only design and structure",
    )

    manager = Agent(
        name="Project Manager",
        instructions="You are a project manager. Asign the work to specialised agent. You job is to get the idea and assign to that specialized agent and in the end ask reviewer to review that and ask it to produce final document. You have to return a final markdown with all the things. Never ask for any confirmation or anything back from user because user can provide input only once so get the idea and generate and if not a valid idea then return a message to user that the idea is not valid and cannot be processed. No thinking or anything extra should be shown to end user, only the final output is to be shown to user which is the final markdown. Run review only once and After getting all the data and review formulate a professional refined document and return that one.",
        model=model,
        tools=[
            requirements_agent.as_tool(
                tool_name="get_requirements",
                tool_description="Get the requirements.",
            ),
            architecture_agent.as_tool(
                tool_name="get_architecture",
                tool_description="Get the architecture.",
            ),
            database_agent.as_tool(
                tool_name="get_database_design",
                tool_description="Get the database design.",
            ),
            api_agent.as_tool(
                tool_name="get_api_design",
                tool_description="Get the API design.",
            ),
            frontend_agent.as_tool(
                tool_name="get_frontend_design",
                tool_description="Get the frontend design.",
            ),
            reviewer_agent.as_tool(
                tool_name="review_document",
                tool_description="Review the generated document.",
            ),
        ],
    )

    # UI
    idea = ui.input(label="Idea")

    spinner = ui.spinner(size="2rem")
    spinner.visible = False

    status = ui.label("Working...")
    status.visible = False

    async def generate():
        print("Generate started")
        spinner.visible = True
        status.visible = True

        idea.visible = False
        generate_button.visible = False

        try:
            print("Calling Runner.run()")
            res = await Runner.run(manager, idea.value)

            result.set_content(res.final_output)

            spinner.visible = False
            reset_button.visible = True
            status.visible = False
        except Exception as e:
            print(f"Error occurred: {e}")
            ui.notify(f"Error: {e}")
            reset()

    def reset():
        spinner.visible = False
        status.visible = False
        idea.visible = True
        generate_button.visible = True
        reset_button.visible = False
        result.set_content("")

    generate_button = ui.button("Generate", on_click=generate)
    reset_button = ui.button("Reset", on_click=reset)

    reset_button.visible = False

    result = ui.markdown("")


ui.run()
