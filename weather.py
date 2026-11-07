
import os
import json
import requests
from dotenv import load_dotenv
from groq import Groq

# Load API keys
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

MODEL = "openai/gpt-oss-20b"


# Weather Tool
def get_weather(city):

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }

    response = requests.get(url, params=params, timeout=15)
    data = response.json()

    if response.status_code == 200:
        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]

        return (
            f"Weather in {city}: {temperature}°C, "
            f"{description}, Humidity: {humidity}%"
        )

    return f"Weather API error: {data.get('message', 'Unknown error')}"


# Available Tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city"
                    }
                },
                "required": ["city"]
            }
        }
    }
]


# System Prompt
system_prompt = """
You are a helpful AI assistant.

Follow the Start, Plan, Action, Observe and Output process.

1. Start: Understand the user query.
2. Plan: Decide which tool is required.
3. Action: Call the relevant tool.
4. Observe: Examine the tool output.
5. Output: Give the final answer.

Use get_weather whenever the user asks about weather.
Do not invent weather information.
"""


messages = [
    {"role": "system", "content": system_prompt}
]


# Agent Loop
while True:

    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if question.lower() == "clear":
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        print("Chat cleared!")
        continue

    if not question:
        continue

    messages.append({
        "role": "user",
        "content": question
    })

    try:

        # Ask LLM to plan and select a tool
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=1000
        )

        message = response.choices[0].message

        # If LLM selects a tool
        if message.tool_calls:

            messages.append(
                message.model_dump(exclude_none=True)
            )

            for tool_call in message.tool_calls:

                function_name = tool_call.function.name
                arguments = json.loads(
                    tool_call.function.arguments
                )

                print("\nStep: Action")
                print("Function:", function_name)
                print("Input:", arguments)

                # Execute selected function
                if function_name == "get_weather":
                    result = get_weather(arguments["city"])
                else:
                    result = "Unknown tool."

                print("\nStep: Observe")
                print("Output:", result)

                # Send tool result back to LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Generate final response
            final_response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_completion_tokens=1000
            )

            answer = final_response.choices[0].message.content

            print("\nStep: Output")
            print("Assistant:", answer)

            messages.append({
                "role": "assistant",
                "content": answer
            })

        else:

            answer = message.content or ""
            print("\nAssistant:", answer)

            messages.append({
                "role": "assistant",
                "content": answer
            })

    except Exception as e:

        messages.pop()
        print("\nError:", type(e).__name__, e)
