from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv()

openai = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
model="@cf/mistralai/mistral-small-3.1-24b-instruct"

# Generate Business Field
messages = [
    {
        "role": "user",
        "content": "Pick A random business field in which we can integerate agentic AI. Avoid repeating similar fields. Return only the business name"
    }
]

biz_field_res= openai.chat.completions.create(model=model, messages=messages)
biz_field=biz_field_res.choices[0].message.content

print("Business Field:", biz_field_res.choices[0].message.content)

# Generate Challenges
messages = [
    {
        "role": "user",
        "content": f"Tell me the challenges faced in this field: {biz_field}, that agentic AI can solve. Just give me a list and nothing extra"
    }
]

challenges_res = openai.chat.completions.create(model=model, messages=messages)
challenges=challenges_res.choices[0].message.content

print(f"Challenges:\n{challenges}")

# Generate Solutions
messages = [
    {
        "role": "user",
        "content": """You have a business field and some problems in it. Now your job is to propose Agentic AI solutions for those problems. Give me jus final solutions and nothing extra"""
    }
]

sol_res = openai.chat.completions.create(model=model, messages=messages)
sol=sol_res.choices[0].message.content

print(f"Solutions:\n{sol}")