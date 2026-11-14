import os
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SIMPLE_MODEL = "openai/gpt-oss-20b"
CODING_MODEL = "openai/gpt-oss-120b"


def ask(model: str, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


# 1. STATE
class State(TypedDict):
    user_query: str
    query_type: str      # "coding" or "simple"
    answer: str
    model_used: str


# 2. NODES
def detect_query(state: State) -> dict:
    system = (
        "Classify the user's question into exactly one word.\n"
        "Reply 'coding' if it is about writing, debugging, explaining or "
        "designing code/software.\n"
        "Reply 'simple' for anything else (general knowledge, chat, facts, "
        "advice).\n"
        "Reply with only the single word."
    )
    reply = ask(SIMPLE_MODEL, system, state["user_query"]).strip().lower()
    query_type = "coding" if "coding" in reply else "simple"
    print(f"[detect_query] -> {query_type}")
    return {"query_type": query_type}


def solve_coding_question(state: State) -> dict:
    system = ("You are an expert programmer. Give correct, clean code "
              "with a short explanation.")
    answer = ask(CODING_MODEL, system, state["user_query"])
    return {"answer": answer, "model_used": CODING_MODEL}


def solve_simple_question(state: State) -> dict:
    system = "You are a helpful assistant. Answer briefly and clearly."
    answer = ask(SIMPLE_MODEL, system, state["user_query"])
    return {"answer": answer, "model_used": SIMPLE_MODEL}


# 3. ROUTER EDGE
def route_query(state: State) -> Literal["solve_coding_question", "solve_simple_question"]:
    if state["query_type"] == "coding":
        return "solve_coding_question"
    return "solve_simple_question"


# 4. BUILD THE GRAPH
builder = StateGraph(State)

builder.add_node("detect_query", detect_query)
builder.add_node("solve_coding_question", solve_coding_question)
builder.add_node("solve_simple_question", solve_simple_question)

builder.add_edge(START, "detect_query")
builder.add_conditional_edges("detect_query", route_query)
builder.add_edge("solve_coding_question", END)
builder.add_edge("solve_simple_question", END)

graph = builder.compile()


# 5. RUN
if __name__ == "__main__":
    print("LangGraph router (type 'exit' to quit)\n")
    while True:
        q = input("Ask your question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        result = graph.invoke({"user_query": q})
        print(f"\nModel used : {result['model_used']}")
        print(f"Answer     :\n{result['answer']}\n")
