"""
LLM Router - GenAI Class 7 Homework
-----------------------------------
Idea: we have many models (name, type, cost). The user asks a question.
A cheap "router" LLM (running on Groq) reads the question + model catalog
and picks the CHEAPEST model that can still do the job well.
It returns: model name + reason.

Groq retires models often, so at startup this script asks Groq which
models YOUR account can use, and only keeps those in the catalog.

Setup:
    pip install groq python-dotenv
    .env file (same folder):  GROQ_API_KEY=gsk_your_key_here
Run:
    python llm_router.py
"""

import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    sys.exit(
        "GROQ_API_KEY not found.\n"
        "Create a file named .env next to llm_router.py containing:\n"
        "GROQ_API_KEY=gsk_your_key_here"
    )

client = Groq(api_key=API_KEY)

# ---------------------------------------------------------------------------
# 1. MODEL CATALOG (candidates)
#    kind = "chat"  -> text models, priced per 1M tokens (cost_in / cost_out)
#    kind = "audio" -> speech-to-text models, priced per hour of audio
#    complexity: 1 = easy tasks only ... 5 = can handle the hardest tasks
#    Prices = Groq late 2026, recheck on console.groq.com/docs/models.
#    Models not available on your Groq account are removed automatically.
# ---------------------------------------------------------------------------
CANDIDATE_MODELS = [
    {
        "name": "openai/gpt-oss-20b",
        "kind": "chat",
        "type": "reasoning (medium)",
        "cost_in": 0.075,
        "cost_out": 0.30,
        "complexity": 4,
        "good_for": "general chat, logic, math, medium coding, step by step reasoning",
    },
    {
        "name": "openai/gpt-oss-120b",
        "kind": "chat",
        "type": "reasoning (large)",
        "cost_in": 0.15,
        "cost_out": 0.60,
        "complexity": 5,
        "good_for": "very hard math, complex coding/system design, deep multi-step analysis",
    },
    {
        "name": "qwen/qwen3.8-27b",
        "kind": "chat",
        "type": "multilingual / vision / creative (preview)",
        "cost_in": 0.80,
        "cost_out": 4.00,
        "complexity": 5,
        "good_for": "Hindi/Hinglish and other languages, image understanding, long creative writing",
    },
    {
        "name": "openai/gpt-oss-safeguard-20b",
        "kind": "chat",
        "type": "safety / content moderation",
        "cost_in": 0.075,
        "cost_out": 0.30,
        "complexity": 2,
        "good_for": "ONLY for checking if text is safe/toxic/policy-violating, moderation, classification",
    },
    {
        "name": "allam-2-7b",
        "kind": "chat",
        "type": "small Arabic + English model",
        "cost_in": 0.05,   # ASSUMED - price not confirmed, edit if you know it
        "cost_out": 0.10,  # ASSUMED
        "complexity": 2,
        "good_for": "Arabic questions, simple English Q&A, short answers",
    },
    {
        "name": "whisper-large-v3-turbo",
        "kind": "audio",
        "type": "speech-to-text (fast, cheap)",
        "cost_text": "$0.04 per hour of audio",
        "complexity": 3,
        "good_for": "transcribing audio/voice/speech files to text (cheapest option)",
    },
    {
        "name": "whisper-large-v3",
        "kind": "audio",
        "type": "speech-to-text (most accurate)",
        "cost_text": "$0.111 per hour of audio",
        "complexity": 5,
        "good_for": "transcribing difficult audio: noisy, accented, many languages, translation to English",
    },
]

# Router must be a chat model. First available one is used.
ROUTER_PREFERENCE = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]


def avg_cost(m: dict) -> float:
    """Single number to compare CHAT models: average of input and output price."""
    return (m["cost_in"] + m["cost_out"]) / 2


def load_available_models():
    """Ask Groq which models this account can use, filter our catalog."""
    try:
        available_ids = {m.id for m in client.models.list().data}
    except Exception as e:
        sys.exit(f"Could not fetch model list from Groq: {e}\nCheck your API key / internet.")

    models = [m for m in CANDIDATE_MODELS if m["name"] in available_ids]
    router = next((n for n in ROUTER_PREFERENCE if n in available_ids), None)

    if not models or router is None:
        print("None of my catalog models are available on your account.")
        print("Models your account CAN use:")
        for mid in sorted(available_ids):
            print("  -", mid)
        sys.exit("Copy some of these names into CANDIDATE_MODELS and run again.")

    return models, router


MODELS, ROUTER_MODEL = load_available_models()
CHAT_MODELS = [m for m in MODELS if m["kind"] == "chat"]


def price_text(m: dict) -> str:
    if m["kind"] == "audio":
        return m["cost_text"]
    return f"${m['cost_in']} in / ${m['cost_out']} out per 1M tokens"


def catalog_as_text() -> str:
    lines = []
    for m in MODELS:
        lines.append(
            f"- name: {m['name']} | kind: {m['kind']} | type: {m['type']} "
            f"| cost: {price_text(m)} | max complexity: {m['complexity']}/5 "
            f"| good for: {m['good_for']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. ROUTER FUNCTION
# ---------------------------------------------------------------------------
def route(query: str) -> dict:
    system_prompt = f"""You are an LLM router. Pick the best model for the user's query.

Rules:
1. Judge the query difficulty (1-5) and the task type (chat, code, math/reasoning,
   writing, translation, moderation, non-English language, audio transcription).
2. Choose the CHEAPEST model whose kind, type and max complexity can handle it well.
3. kind "audio" models ONLY transcribe audio. kind "chat" models can NOT process audio files.
   Use an audio model only when the user wants audio/speech transcribed.
4. Use a specialised model (moderation, multilingual, Arabic) only when the task matches it.
5. Never pick an expensive model for an easy query.
6. If NO model in the list can do the task, answer with model "NONE".
7. Only choose from the models listed below.

Available models:
{catalog_as_text()}

Reply ONLY with JSON in this exact format:
{{"model": "<model name or NONE>", "difficulty": <1-5>, "reason": "<1-2 sentences why this model, mention cost and fit>"}}"""

    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        if result.get("model") == "NONE":
            return result

        valid_names = [m["name"] for m in MODELS]
        if result.get("model") not in valid_names:
            raise ValueError("Router picked an unknown model")

        # Safety net (chat models only): upgrade if difficulty exceeds the model's limit
        chosen = next(m for m in MODELS if m["name"] == result["model"])
        difficulty = result.get("difficulty")
        if chosen["kind"] == "chat" and isinstance(difficulty, int) and difficulty > chosen["complexity"]:
            stronger = [m for m in CHAT_MODELS if m["complexity"] >= difficulty]
            if stronger:
                upgrade = min(stronger, key=avg_cost)
                result["reason"] += (
                    f" [Upgraded from {chosen['name']}: difficulty {difficulty} "
                    f"exceeds its limit of {chosen['complexity']}]"
                )
                result["model"] = upgrade["name"]

        return result

    except Exception as e:
        err = str(e)

        # Router refused the request / returned no JSON -> do NOT forward it anywhere
        if "json_validate_failed" in err:
            return {
                "model": "BLOCKED",
                "difficulty": "n/a",
                "reason": "Request was refused by the router (unsafe or invalid). "
                          "It is not forwarded to any model.",
            }

        # Other errors (network, rate limit...) -> fall back to cheapest CHAT model
        cheapest = min(CHAT_MODELS, key=avg_cost)
        return {
            "model": cheapest["name"],
            "difficulty": "unknown",
            "reason": f"Fallback to cheapest model because routing failed ({e}).",
        }


# ---------------------------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Router model : {ROUTER_MODEL}")
    print("Models in use:")
    for m in MODELS:
        print(f"  - {m['name']}  ({m['type']}, {price_text(m)})")
    print("\nLLM Router (type 'exit' to quit)\n")

    while True:
        q = input("Ask your question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        out = route(q)
        print(f"\n  Model      : {out['model']}")
        print(f"  Difficulty : {out['difficulty']}")
        print(f"  Why        : {out['reason']}\n")