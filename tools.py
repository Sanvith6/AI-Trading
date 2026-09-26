"""
Agent Tools Module
Provides practical system, file, python execution, and web tools for the AI Agent.
Includes OpenAI-compatible schemas for function calling.
"""

import os
import sys
import subprocess
import urllib.request
import json
import re
from typing import Dict, Any, List, Optional

WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))

def _resolve_path(path: str) -> str:
    """Resolve relative path against the workspace directory."""
    if not os.path.isabs(path):
        return os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    return os.path.abspath(path)

# --- 1. Tool Implementations ---

def execute_command(command: str, timeout: int = 30) -> str:
    """Execute a system command via PowerShell on Windows and return stdout/stderr."""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout
        err = proc.stderr
        ret = f"Exit code: {proc.returncode}\n"
        if out:
            ret += f"Output:\n{out}\n"
        if err:
            ret += f"Error:\n{err}\n"
        return ret.strip() or "[Command executed with no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

def read_file(path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read contents of a file. Optionally specify start_line and end_line (1-indexed)."""
    target = _resolve_path(path)
    if not os.path.exists(target):
        return f"Error: File not found at '{path}'."
    if os.path.isdir(target):
        return f"Error: '{path}' is a directory, not a file. Use list_directory instead."

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total = len(lines)
        if start_line is not None or end_line is not None:
            s = max(1, start_line or 1)
            e = min(total, end_line or total)
            selected = lines[s - 1 : e]
            return f"Lines {s} to {e} of {total} in '{path}':\n" + "".join(selected)
        else:
            return "".join(lines) or "[File is empty]"
    except Exception as e:
        return f"Error reading file '{path}': {str(e)}"

def write_file(path: str, content: str, append: bool = False) -> str:
    """Create or overwrite a file with the specified content."""
    target = _resolve_path(path)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(content)
        action = "appended to" if append else "written to"
        size = os.path.getsize(target)
        return f"Successfully {action} '{path}' ({size} bytes)."
    except Exception as e:
        return f"Error writing to file '{path}': {str(e)}"

def list_directory(path: str = ".") -> str:
    """List files and directories within a given path."""
    target = _resolve_path(path)
    if not os.path.exists(target):
        return f"Error: Path '{path}' does not exist."
    if not os.path.isdir(target):
        return f"Error: '{path}' is a file, not a directory."

    try:
        items = []
        for entry in os.scandir(target):
            info = {
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": entry.stat().st_size if entry.is_file() else None,
            }
            items.append(info)
        return json.dumps(items, indent=2)
    except Exception as e:
        return f"Error listing directory '{path}': {str(e)}"

def search_files(pattern: str, path: str = ".", regex: bool = False) -> str:
    """Search for text or regex pattern across files in the directory."""
    target = _resolve_path(path)
    matches = []
    try:
        flags = 0 if regex else re.IGNORECASE
        compiled_pattern = re.compile(pattern if regex else re.escape(pattern), flags)

        for root, _, files in os.walk(target):
            # Skip virtual environments and git folders
            if any(part.startswith((".", "node_modules", "venv", "__pycache__")) for part in root.split(os.sep)):
                continue
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f, 1):
                            if compiled_pattern.search(line):
                                rel_path = os.path.relpath(filepath, WORKSPACE_DIR)
                                matches.append(f"{rel_path}:{line_idx}: {line.strip()}")
                                if len(matches) >= 30:
                                    matches.append("... [Search truncated at 30 results]")
                                    return "\n".join(matches)
                except Exception:
                    continue

        if not matches:
            return f"No matches found for '{pattern}' in '{path}'."
        return "\n".join(matches)
    except Exception as e:
        return f"Error searching files: {str(e)}"

def run_python(code: str, timeout: int = 20) -> str:
    """Execute Python code in an isolated subprocess and return stdout/stderr.

    SECURITY: This tool executes arbitrary Python code. It is classified as a
    sensitive tool and requires explicit confirmation in safe mode (see agent.py
    `is_sensitive` list). Sandboxing is limited to subprocess timeout + cwd.
    Do NOT remove this from the sensitive list.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout
        err = proc.stderr
        res = ""
        if out:
            res += f"Output:\n{out}\n"
        if err:
            res += f"Error / Stderr:\n{err}\n"
        if proc.returncode != 0:
            res += f"Process exited with code {proc.returncode}\n"
        return res.strip() or "[Executed successfully with no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Python execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error running Python code: {str(e)}"

def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch webpage content or JSON API data via HTTP GET request."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AI-Agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            # Strip extra HTML tags if it's HTML for readable text
            text = re.sub(r"<script.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > max_chars:
                return text[:max_chars] + f"\n... [Truncated {len(text) - max_chars} characters]"
            return text or "[Empty response]"
    except Exception as e:
        return f"Error fetching URL '{url}': {str(e)}"


# --- 2. Tool Registry & Schemas ---

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Execute a shell command (PowerShell on Windows) in the workspace directory. Use this to run scripts, build projects, check git status, test code, or manage system utilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run (e.g. 'dir', 'git status', 'python script.py', 'pip list')",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max execution time in seconds (default 30)",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file in the workspace or system. Supports optional line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read (relative to workspace or absolute).",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed starting line number.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed ending line number.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite/append to an existing file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to create or edit.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact content to write to the file.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Set True to append instead of overwrite (default False).",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and folders in a specified directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default is '.' for workspace root).",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for text or regex pattern in workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern or regex to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search within (default is '.').",
                    },
                    "regex": {
                        "type": "boolean",
                        "description": "Whether to treat pattern as regex (default False).",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute arbitrary Python code in a subprocess to perform calculations, test code logic, process data, or verify scripts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code snippet to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch web pages or REST APIs to read external online articles, documentation, or data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The HTTP/HTTPS URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "execute_command": execute_command,
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "search_files": search_files,
    "run_python": run_python,
    "fetch_url": fetch_url,
}

def call_tool(name: str, arguments: Dict[str, Any]) -> str:
    """Safely invoke a registered tool with parsed arguments."""
    if name not in TOOL_REGISTRY:
        return f"Error: Tool '{name}' is not recognized. Available tools: {list(TOOL_REGISTRY.keys())}"
    try:
        fn = TOOL_REGISTRY[name]
        return str(fn(**arguments))
    except Exception as e:
        return f"Error executing tool '{name}': {str(e)}"
