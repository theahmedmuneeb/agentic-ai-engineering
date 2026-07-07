import asyncio
from openai import AsyncOpenAI
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

async_openai = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
openai = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))

prompt="Tell me best joke ever"

messages = [
    {
        "role": "user",
        "content": (
            "You are an AI assistant. Keep responses concise and short and only answer what users asks in a shortest possible way.\n\n"
            f"{prompt}"
        )
    }
]

models = [
    "@cf/zai-org/glm-5.2",
    "@cf/openai/gpt-oss-120b",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/meta/llama-guard-3-8b",
    "@cf/google/gemma-4-26b-a4b-it"
]

judge_model="@cf/moonshotai/kimi-k2.6"

async def generateResponses():
    tasks = []
    
    for model in models:
        tasks.append(
            async_openai.chat.completions.create(
                model=model,
                messages=messages,
            )
        )
    
    responses = await asyncio.gather(*tasks)
    
    return responses

responses = asyncio.run(generateResponses())

# Judge
content = ""
for idx, response in enumerate(responses):
    content+=f"Response {idx}:\n{response.choices[0].message.content}\n"

messages=[
    {
        "role": "user",
        "content": f"You are a judge that will judge different response and pick final one. in the response you will only send the response number eg. 0,1,2,3 etc. Just a number.\n\nActual Prompt: \n{prompt}\n\nResponses:\n{content}"
    }
]

judge_res =  openai.chat.completions.create(model=judge_model, messages=messages) 

winner= int(judge_res.choices[0].message.content)

print("Best Model:", models[winner])
print("Best Response:\n", responses[winner].choices[0].message.content)


