from agents import (
    Agent,
    function_tool,
    SQLiteSession,
    Runner,
    set_tracing_disabled,
    OpenAIChatCompletionsModel,
)
import gradio as gr
from pydantic import BaseModel
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from typing import Generic, TypeVar


load_dotenv()


class ChecklistItem(BaseModel):
    name: str
    checked: bool


class Recipe(BaseModel):
    name: str
    ingredients: list[ChecklistItem]
    steps: list[ChecklistItem]


T = TypeVar("T")


class ToolResponse(BaseModel, Generic[T]):
    success: bool
    message: str | None = None
    data: T | None = None


recipe: Recipe | None = None


# Tools
@function_tool
def save_recipe(
    name: str,
    ingredients: list[ChecklistItem],
    steps: list[ChecklistItem],
) -> ToolResponse:
    """
    Create/Replace the recipe.

    Args:
        name: Short Name of the recipe
        ingredients: A list of ingredients object used in the recipe. Must be an object and contain a "name" (string) and a "checked" (boolean). The "name" is the name/label of ingredient and "checked" represents whether user already has the ingredient
        steps: A list of cooking step objects. Each object must contain "name" (string) and "checked" (boolean). "name" is the name/label of step and "checked" represents whether the step has already completed.

    Returns: An object with Success to either True or False based on whether its saved or not.
    """

    global recipe

    recipe = Recipe(
        name=name,
        ingredients=ingredients,
        steps=steps,
    )

    return ToolResponse(success=True)


@function_tool
def get_recipe() -> ToolResponse[Recipe]:
    """
    get the current recipe
    Returns: An object with Success to either True or False based on whether its saved or not. if success true then recipe data will be in 'data' which will be object with recipe name, ingredients and steps.
    """

    if not recipe:
        return ToolResponse(success=False, message="No active recipe")

    return ToolResponse[Recipe](
        success=True,
        data=recipe,
    )


@function_tool
def update_ingredient(
    name: str,
    checked: bool,
) -> ToolResponse:
    """
    Update the checked status of an ingredient

    Args:
        name: Name of the ingredient
        checked: Whether the ingredient is available/fulfilled

    Retrun an object with success either true/false and a message which may or maynot be present
    """

    if not recipe:
        return ToolResponse(success=False, message="No active recipe")

    for ingredient in recipe.ingredients:
        if ingredient.name.lower() == name.lower():
            ingredient.checked = checked
            return ToolResponse(success=True)

    return ToolResponse(success=False, message=f"Ingredient '{name}' not found")


@function_tool
def update_step(
    name: str,
    checked: bool,
) -> ToolResponse:
    """
    Update the checked status of a step

    Args:
        name: Name of the step
        checked: Whether the step is completed

    Retrun an object with success either true/false and a message which may or maynot be present
    """

    if not recipe:
        return ToolResponse(success=False, message="No active recipe")

    for step in recipe.steps:
        if step.name.lower() == name.lower():
            step.checked = checked
            return ToolResponse(success=True)

    return ToolResponse(success=False, message=f"Step '{name}' not found")


# Agent
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

set_tracing_disabled(True)

agent = Agent(
    name="Bawarchi",
    instructions="Your name is Bawarchi and you are a kitchen AI agent that will help to prepare the meal/recipe. You job is to stay with user till the end until the recipe is prepared. You will have access to the tools for create/modifying recipe. At one time there will be only one recipe so if user demands something new tell him that it will replace the old progress. Stay short concise and best way you can help the user. You dont need to show ingredients specially in every response unless asked. Stay natural. if user ask anything except cooking or recipe deny them and also dont make anything from your self if you dont have any context. When user gathers ingredient or do some step mark that checked. try that first user gathers all ingredients but its not necessary and dont force to do so and then move to steps so when they tell you that thay has these few things or have done any step mark them done and Tell them next. Make sure to not to mark a step unless its required ingredients are fulfilled.",
    tools=[save_recipe, update_ingredient, update_step, get_recipe],
    model=OpenAIChatCompletionsModel(
        model=os.getenv("OPENAI_MODEL"), openai_client=openai_client
    ),
)


# Logic
session = SQLiteSession("kitchen_agent")


async def handle_chat(message, _):
    res = await Runner.run(agent, message, session=session)
    return res.final_output


#
def render_recipe():
    content = ""
    if not recipe:
        content = """## No current active recipe"""
    else:
        content = f"# {recipe.name}\n\n"
        content += "## Ingredients:\n\n"
        for ingredient in recipe.ingredients:
            if ingredient.checked:
                content += "☑"
            else:
                content += "☐"
            content += f" {ingredient.name}<br>"

        content += "\n\n## Steps:\n\n"
        for step in recipe.steps:
            if step.checked:
                content += "☑"
            else:
                content += "☐"
            content += f" {step.name}<br>"

    return content


with gr.Blocks() as ui:
    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                handle_chat,
            )
        with gr.Column(scale=1):
            recipe_view = gr.Markdown()
            timer = gr.Timer(1)
            timer.tick(
                fn=render_recipe,
                outputs=recipe_view,
            )

ui.launch(inbrowser=True)
