"""
Local coding agent (Cursor-style) powered by Ollama.

Tools: list_files, read_file, search_in_files, create_file, edit_file,
       delete_file, make_dir, run_command
All file access is sandboxed to a WORKSPACE folder.
Writes, deletes and commands ask for your approval (type /auto to skip asking).
"""
import os
import re
import sys
import json
import difflib
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
ROOT = Path(os.getenv("AGENT_WORKSPACE", "workspace")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)

IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode"}
MAX_READ_CHARS = 20000
MAX_OUTPUT_CHARS = 4000
MAX_ROUNDS = 12
AUTO_APPROVE = False


# ======================= Ollama server / model =======================
def ensure_server():
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5).raise_for_status()
    except Exception:
        print("Cannot reach Ollama at", OLLAMA_HOST)
        print("Start Docker Desktop, then run:  docker start ollama")
        sys.exit(1)


def model_installed(name):
    data = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).json()
    return any(m["name"] in (name, f"{name}:latest") for m in data.get("models", []))


def pull_model(name):
    print(f"Pulling {name} (first time only)...")
    with requests.post(f"{OLLAMA_HOST}/api/pull", json={"model": name, "stream": True},
                       stream=True, timeout=None) as r:
        if r.status_code != 200:
            print(f"\nPull failed ({r.status_code}): {r.text}")
            sys.exit(1)
        for line in r.iter_lines():
            if not line:
                continue
            info = json.loads(line)
            if "error" in info:
                print("\nPull failed:", info["error"])
                sys.exit(1)
            status = info.get("status", "")
            if info.get("total") and info.get("completed"):
                status += f": {info['completed'] * 100 // info['total']}%"
            print("\r" + status.ljust(60), end="", flush=True)
    print("\nModel ready.\n")


# ======================= Helpers =======================
def safe_path(rel):
    """Resolve a path and make sure it stays inside the workspace."""
    p = (ROOT / rel).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError(f"Path '{rel}' is outside the workspace.")
    return p


def rel(p):
    return str(p.relative_to(ROOT)).replace("\\", "/") or "."


def truncate(text, limit=MAX_OUTPUT_CHARS):
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def approve(action, preview=""):
    if AUTO_APPROVE:
        return True
    print(f"\n>> {action}")
    if preview:
        print(truncate(preview, 2500))
    return input("Approve? [y/N]: ").strip().lower() in ("y", "yes")


def make_diff(old, new, name):
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(),
                                f"a/{name}", f"b/{name}", lineterm="")
    return "\n".join(diff)


# ======================= Tools =======================
def list_files(path="."):
    base = safe_path(path)
    if not base.exists():
        return f"Error: '{path}' does not exist."
    if base.is_file():
        return rel(base)
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        for f in sorted(filenames):
            out.append(rel(Path(dirpath) / f))
        if len(out) > 300:
            out.append("... (too many files, list a subfolder)")
            break
    return "\n".join(out) or "(empty folder)"


def read_file(path):
    p = safe_path(path)
    if not p.is_file():
        return f"Error: file '{path}' not found."
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "Error: not a UTF-8 text file."
    numbered = "\n".join(f"{i + 1}| {line}" for i, line in enumerate(text.splitlines()))
    return truncate(numbered, MAX_READ_CHARS) or "(empty file)"


def search_in_files(pattern, path="."):
    base = safe_path(path)
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    hits = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for f in filenames:
            fp = Path(dirpath) / f
            try:
                for n, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{rel(fp)}:{n}: {line.strip()}")
            except (UnicodeDecodeError, OSError):
                continue
    return truncate("\n".join(hits[:100])) or "No matches."


def create_file(path, content):
    p = safe_path(path)
    exists = p.exists()
    if exists and not p.is_file():
        return f"Error: '{path}' is a folder."
    old = p.read_text(encoding="utf-8") if exists else ""
    label = "OVERWRITE" if exists else "CREATE"
    preview = make_diff(old, content, path) if exists else content
    if not approve(f"{label} {path}", preview):
        return "User rejected this change."
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"{'Overwrote' if exists else 'Created'} {path} ({len(content.splitlines())} lines)."


def edit_file(path, old_str, new_str):
    p = safe_path(path)
    if not p.is_file():
        return f"Error: file '{path}' not found."
    text = p.read_text(encoding="utf-8")
    count = text.count(old_str)
    if count == 0:
        return "Error: old_str not found. Use read_file and copy the exact text."
    if count > 1:
        return f"Error: old_str appears {count} times. Include more surrounding lines to make it unique."
    new_text = text.replace(old_str, new_str, 1)
    if not approve(f"EDIT {path}", make_diff(text, new_text, path)):
        return "User rejected this change."
    p.write_text(new_text, encoding="utf-8")
    return f"Edited {path}."


def delete_file(path):
    p = safe_path(path)
    if p == ROOT:
        return "Error: cannot delete the workspace root."
    if not p.is_file():
        return f"Error: file '{path}' not found (only files can be deleted)."
    if not approve(f"DELETE {path}"):
        return "User rejected this change."
    p.unlink()
    return f"Deleted {path}."


def make_dir(path):
    p = safe_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return f"Folder ready: {path}"


def run_command(command):
    if not approve(f"RUN in workspace: {command}"):
        return "User rejected this command."
    try:
        r = subprocess.run(command, shell=True, cwd=ROOT, capture_output=True,
                           text=True, timeout=60)
        out = (r.stdout or "") + (r.stderr or "")
        return truncate(f"exit code {r.returncode}\n{out}")
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds."


TOOL_FUNCS = {
    "list_files": list_files, "read_file": read_file, "search_in_files": search_in_files,
    "create_file": create_file, "edit_file": edit_file, "delete_file": delete_file,
    "make_dir": make_dir, "run_command": run_command,
}


def tool(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = lambda d: {"type": "string", "description": d}

TOOLS = [
    tool("list_files", "List all files in a folder (recursive).", {"path": S("Folder, default '.'")}, []),
    tool("read_file", "Read a file with line numbers.", {"path": S("File path")}, ["path"]),
    tool("search_in_files", "Search text/regex across files.",
         {"pattern": S("Text or regex"), "path": S("Folder, default '.'")}, ["pattern"]),
    tool("create_file", "Create a new file or fully overwrite one.",
         {"path": S("File path"), "content": S("Full file content")}, ["path", "content"]),
    tool("edit_file", "Replace one exact, unique snippet in a file.",
         {"path": S("File path"), "old_str": S("Exact existing text, unique in the file"),
          "new_str": S("Replacement text")}, ["path", "old_str", "new_str"]),
    tool("delete_file", "Delete a file.", {"path": S("File path")}, ["path"]),
    tool("make_dir", "Create a folder.", {"path": S("Folder path")}, ["path"]),
    tool("run_command", "Run a shell command in the workspace (e.g. run tests or a script).",
         {"command": S("Shell command")}, ["command"]),
]

SYSTEM_PROMPT = f"""You are a coding agent working inside the folder '{ROOT.name}'.
You can create, read, update and delete files using your tools.

Rules:
- Use relative paths only.
- Before editing a file, read it first. For small changes use edit_file with an exact, unique old_str.
- Use create_file only for new files or complete rewrites.
- To understand a project, use list_files and search_in_files.
- After changing code, you may run it with run_command to verify it.
- If a tool returns an error, fix your input and try again.
- Never invent file contents. Keep answers short. When done, summarise what you changed."""


# ======================= LLM call =======================
def chat(messages):
    r = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={"model": MODEL, "messages": messages, "tools": TOOLS, "stream": False,
              "think": False, "keep_alive": "30m", "options": {"num_ctx": 8192}},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["message"]


def clean(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).split("</think>")[-1].strip()


# ======================= Agent loop =======================
def main():
    global AUTO_APPROVE
    ensure_server()
    if not model_installed(MODEL):
        pull_model(MODEL)

    print(f"Model: {MODEL}\nWorkspace: {ROOT}")
    print("Commands: exit | clear | /auto (toggle auto-approve)\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if question.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Chat cleared!")
            continue
        if question.lower() == "/auto":
            AUTO_APPROVE = not AUTO_APPROVE
            print("Auto-approve:", "ON" if AUTO_APPROVE else "OFF")
            continue

        messages.append({"role": "user", "content": question})
        turn_start = len(messages) - 1

        try:
            for _ in range(MAX_ROUNDS):
                message = chat(messages)
                message["content"] = clean(message.get("content"))
                messages.append(message)

                calls = message.get("tool_calls")
                if not calls:
                    print("\nAgent:", message["content"])
                    break

                for call in calls:
                    name = call["function"]["name"]
                    args = call["function"]["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args or "{}")

                    print(f"\n[tool] {name}({', '.join(f'{k}={str(v)[:40]!r}' for k, v in args.items())})")
                    func = TOOL_FUNCS.get(name)
                    try:
                        result = func(**args) if func else f"Error: unknown tool '{name}'."
                    except TypeError as e:
                        result = f"Error: bad arguments: {e}"
                    except Exception as e:
                        result = f"Error: {type(e).__name__}: {e}"

                    print("   ->", truncate(str(result), 200).replace("\n", "\n      "))
                    messages.append({"role": "tool", "tool_name": name, "content": str(result)})
            else:
                print("\nAgent: Stopped after too many steps. Ask me to continue.")
        except KeyboardInterrupt:
            print("\n(interrupted)")
            del messages[turn_start:]
        except Exception as e:
            del messages[turn_start:]
            print("\nError:", type(e).__name__, e)


if __name__ == "__main__":
    main()