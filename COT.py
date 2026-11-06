
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

system_prompt = """You are an expert mathematical and general-purpose assistant. Explain answers in simple language. For every question, follow these five steps: ANALYSIS, THINK, OUTPUT, VALIDATE, RESULT. For mathematics, apply BODMAS correctly. Example: 2 + 2 * 10 = 22, because multiplication comes first. For general questions, provide a relevant explanation. Be accurate, concise, and never invent facts. Return plain text with all five headings. Do not use JSON or call tools."""

messages = [{"role": "system", "content": system_prompt}]

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if question.lower() == "clear":
        messages = [{"role": "system", "content": system_prompt}]
        print("Chat cleared!")
        continue

    if not question:
        continue

    messages.append({"role": "user", "content": question})

    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=1200
        )

        answer = result.choices[0].message.content

        print("\nAssistant:\n")
        print(answer)

        messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        messages.pop()
        print(f"\nError: {type(e).__name__}: {e}")
