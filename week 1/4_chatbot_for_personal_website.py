import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
from pypdf import PdfReader
import csv
import json

load_dotenv()

openai = OpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
model="@cf/moonshotai/kimi-k2.6"

profile_pdf_reader = PdfReader("week 1/assets/profile.pdf")

profile_details=""
for page in profile_pdf_reader.pages:
    profile_details += page.extract_text() + "\n"
    
with open("week 1/assets/details.txt", "r", encoding="utf-8") as file:
    additional_profile_details = file.read()
    
system_prompt=f"""You are an AI chatbot for a personal portfolio website for a person. You have to reply to the conversations/questions on his behalf. No extra huge content or descriptions, keep short and concise. Dont answer bindly. Only answer from details provided and you can say "I dont know if you dont have the context. Use a natural tone not like a bot and only answer related to that person. You are not that person you are actually an assistant of them." 

# Person Details:
{profile_details}

# Additional Details:
{additional_profile_details}
"""

tools = [
    {
        "type": "function",
        "function": {
            "name": "subscribe_newsletter",
            "description": "use this tool to subscribe someone to newsletter",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Subscriber name"
                    },
                    "email": {
                        "type": "string",
                        "description": "Email address to subscribe"
                    }
                },
                "required": ["email", "name"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_resume",
            "description": "This tool give the resume url",
        }
    }
]

def handle_subscribe_newsletter(name, email):
    csv_path="week 1/assets/newsletter.csv"
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "email"])
        
        if os.path.getsize(csv_path) == 0:
            writer.writeheader()
            
        writer.writerow({
            "name": name,
            "email": email
        })
    
    return {
        "success": True
    }
    
def handle_get_resume():
    return {
        "success": True,
        "url": "https://example.com/resume.pdf"
    }

tool_box = {
    "subscribe_newsletter": handle_subscribe_newsletter,
    "get_resume": handle_get_resume,
}

def chat(message, history):
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    
    res = openai.chat.completions.create(model=model, messages=messages, tools=tools)
    
    while res.choices[0].finish_reason == "tool_calls":
        message = res.choices[0].message
        tool_calls = message.tool_calls
        
        results=[]
        
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = json.loads(tool_call.function.arguments)
            
            tool=tool_box[tool_name]
            
            result=""
            
            if tool:
                result = tool(**tool_arguments)
            else:
                result= "Tool Not found"
                
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        
        messages.append(message)
        messages.extend(results)
        
        res = openai.chat.completions.create(model=model, messages=messages, tools=tools)
    
    return res.choices[0].message.content

gr.ChatInterface(chat).launch(inbrowser=True)