from agents.mcp import MCPServerStreamableHttp
import os
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    input_guardrail,
    GuardrailFunctionOutput,
    set_tracing_disabled,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import gradio as gr

load_dotenv()

# OpenAI client and agnet model
openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
model = OpenAIChatCompletionsModel(
    model=os.getenv("OPENAI_MODEL"), openai_client=openai_client
)
set_tracing_disabled(True)

# MCP Servers
exa = MCPServerStreamableHttp(
    params={
        "url": "https://mcp.exa.ai/mcp",
        "headers": {
            "Authorization": f"Bearer {os.getenv('EXA_API_KEY')}",
        },
    },
)


# Pydantic models
class ValidationAgentOutput(BaseModel):
    is_valid: bool = Field(
        ..., description="Whether the research topic is valid or not"
    )
    reason: str = Field(
        None, description="If the topic is invalid, provide a reason for the invalidity"
    )


# Guardrails
@input_guardrail
async def research_guardrail(ctx, agent, input):
    result = await Runner.run(
        research_guardrail_agent,
        input,
        context=ctx.context,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output.reason,
        tripwire_triggered=not result.final_output.is_valid,
    )


# Agents
research_guardrail_agent = Agent(
    name="Validation Agent",
    instructions="You are a validator. Validate that the request contains a research topic.",
    output_type=ValidationAgentOutput,
    model=model,
)

research_agent = Agent(
    name="Research Assistant",
    instructions="Research the user's topic using Exa and produce a clear, well-structured summary with headings, key findings, references. Do an extensive search and provide a comprehensive summary. You can search the web and use Exa to find relevant information.",
    mcp_servers=[exa],
    input_guardrails=[research_guardrail],
    model=model,
)


async def handle_chat(message, _):
    async with exa:
        result = await Runner.run(
            research_agent,
            message,
        )

        return result.final_output


ui = gr.ChatInterface(
    fn=handle_chat,
    title="Research Agent",
)

ui.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
)
