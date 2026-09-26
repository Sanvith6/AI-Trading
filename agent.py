"""
AI Agent Engine
Autonomous multi-step reasoning & tool execution loop powered by OpenRouter.
"""

import json
import os
from typing import List, Dict, Any, Optional, Callable, Generator
from openai import OpenAI
from dotenv import load_dotenv

import tools
from llm_client import get_client, DEFAULT_MODEL

load_dotenv(override=True)

DEFAULT_SYSTEM_PROMPT = """You are an autonomous AI Agent designed to help the user accomplish complex coding and system tasks.
You have access to a set of real tools allowing you to:
1. Run shell commands (execute_command)
2. Read files (read_file)
3. Create or modify files (write_file)
4. List directory contents (list_directory)
5. Search across files (search_files)
6. Run Python scripts and calculations (run_python)
7. Fetch external URLs / web data (fetch_url)

Guidelines:
- When given a task, break it down logically.
- Use your tools proactively to inspect files, gather facts, verify work, and run code.
- If you need to make changes to a file, first read the file if you don't know its contents.
- Verify your changes or created scripts by running them or checking outputs.
- Be clear, precise, and concise. Explain what actions you took and summarize the outcome.
"""

class AgentEvent:
    """Represents an event during agent execution."""
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.type = event_type
        self.data = data

    def __repr__(self):
        return f"<AgentEvent type={self.type} data={self.data}>"


class Agent:
    """
    Autonomous ReAct (Reasoning + Acting) Agent.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_steps: int = 15,
        safe_mode: bool = False,
        confirm_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        temperature: float = 0.4,
    ):
        self.model = model or DEFAULT_MODEL
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_steps = max_steps
        self.safe_mode = safe_mode
        self.confirm_callback = confirm_callback
        self.temperature = temperature
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def reset(self, system_prompt: Optional[str] = None) -> None:
        """Reset conversation history while retaining system prompt."""
        if system_prompt:
            self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_tool_defs(self) -> List[Dict[str, Any]]:
        """Return OpenAI tool schemas. Subclasses (e.g. TradingAgent) override this."""
        return tools.TOOL_DEFINITIONS

    def call_tool_fn(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool by name. Subclasses may override to add domain tools."""
        return tools.call_tool(name, arguments)

    def step_stream(self, user_instruction: str) -> Generator[AgentEvent, None, str]:
        """
        Execute the agent loop for a user instruction, yielding AgentEvent objects in real time.
        Returns the final response string.
        """
        client = get_client()
        self.messages.append({"role": "user", "content": user_instruction})
        yield AgentEvent("user_message", {"content": user_instruction})

        step = 0
        final_answer = ""

        while step < self.max_steps:
            step += 1
            yield AgentEvent("step_start", {"step": step, "max_steps": self.max_steps})

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.get_tool_defs(),
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=2048,
                )
            except Exception as e:
                err_msg = f"Model API error: {str(e)}"
                yield AgentEvent("error", {"message": err_msg})
                return err_msg

            choice = response.choices[0]
            message = choice.message

            # Check if model provided thought/text content
            if message.content:
                yield AgentEvent("thought", {"content": message.content})

            # Check for tool calls
            tool_calls = message.tool_calls or []

            # If no tools called, this is the final answer!
            if not tool_calls:
                final_answer = message.content or "[Task completed with no final text response]"
                self.messages.append({"role": "assistant", "content": final_answer})
                yield AgentEvent("final_answer", {"content": final_answer})
                return final_answer

            # Record assistant's tool-calling message into history
            self.messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # Execute each tool call
            for tc in tool_calls:
                call_id = tc.id
                fn_name = tc.function.name
                fn_args_raw = tc.function.arguments

                try:
                    fn_args = json.loads(fn_args_raw)
                except Exception:
                    fn_args = {}

                yield AgentEvent("tool_start", {
                    "tool_name": fn_name,
                    "arguments": fn_args,
                    "call_id": call_id,
                })

                # Safe mode check for sensitive actions
                is_sensitive = fn_name in ["execute_command", "write_file", "run_python"]
                if self.safe_mode and is_sensitive and self.confirm_callback:
                    approved = self.confirm_callback(fn_name, fn_args)
                    if not approved:
                        result = f"User denied permission to execute tool '{fn_name}' with args {fn_args}."
                        yield AgentEvent("tool_denied", {
                            "tool_name": fn_name,
                            "arguments": fn_args,
                            "call_id": call_id,
                        })
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": result,
                        })
                        continue

                # Execute tool
                result = self.call_tool_fn(fn_name, fn_args)
                yield AgentEvent("tool_end", {
                    "tool_name": fn_name,
                    "result": result,
                    "call_id": call_id,
                })

                # Append tool response to message history
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result,
                })

        # Max steps exceeded
        warning = f"Agent reached maximum execution steps ({self.max_steps})."
        yield AgentEvent("max_steps_reached", {"message": warning})
        return warning

    def run(self, user_instruction: str) -> str:
        """Synchronously execute the agent loop to completion and return final response."""
        final_res = ""
        for event in self.step_stream(user_instruction):
            if event.type == "final_answer":
                final_res = event.data.get("content", "")
        return final_res
