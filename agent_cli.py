"""
Interactive Terminal Interface for the AI Agent
Features rich formatting, live status spinners, tool visualization panels, and safe mode.
Run with: python agent_cli.py
"""

import sys
import json
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from agent import Agent, AgentEvent
import tools
from llm_client import DEFAULT_MODEL

console = Console()

BANNER = rf"""
[bold cyan]╔═══════════════════════════════════════════════════════════════════════════╗
║                      ✨ AUTONOMOUS AI AGENT CONSOLE ✨                    ║
║   Model: [bold green]{DEFAULT_MODEL:<45}[/bold green]║
║   Tools: [yellow]Files, PowerShell, Python Execution, Search, Web Fetch[/yellow]       ║
╚═══════════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""

HELP_TEXT = """
[bold yellow]Available Commands:[/bold yellow]
  [cyan]/tools[/cyan]         - List all tools the agent can use
  [cyan]/safe[/cyan]          - Toggle Safe Mode (require confirmation before running commands/edits)
  [cyan]/clear[/cyan]         - Clear conversation memory and start fresh
  [cyan]/model [name][/cyan]  - View or switch active model
  [cyan]/help[/cyan]          - Show this command reference
  [cyan]/exit[/cyan] or [cyan]exit[/cyan]   - Quit the agent console
"""

def print_tools_table():
    """Render a table of registered agent tools."""
    table = Table(title="🛠️ Registered Agent Tools", border_style="cyan", header_style="bold magenta")
    table.add_column("Tool Name", style="bold green", width=20)
    table.add_column("Description", style="white")

    for td in tools.TOOL_DEFINITIONS:
        fn = td["function"]
        table.add_row(fn["name"], fn["description"].strip())

    console.print(table)

def main():
    console.print(BANNER)
    console.print("[dim]Type your task or question below. Type [bold cyan]/help[/bold cyan] for commands.[/dim]\n")

    safe_mode = False

    def cli_confirm_callback(tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Prompt user in terminal before running sensitive operations."""
        console.print(Panel(
            f"[bold red]⚠️ CONFIRMATION REQUIRED[/bold red]\n"
            f"The agent wants to execute sensitive tool: [bold yellow]{tool_name}[/bold yellow]\n"
            f"Arguments:\n[dim]{json.dumps(arguments, indent=2)}[/dim]",
            border_style="red",
        ))
        return Confirm.ask("[bold yellow]Allow this action?[/bold yellow]", default=True)

    agent = Agent(
        model=DEFAULT_MODEL,
        safe_mode=safe_mode,
        confirm_callback=cli_confirm_callback,
    )

    while True:
        try:
            prompt_str = f"[bold green]Agent ({agent.model.split('/')[-1]})[/bold green]"
            if agent.safe_mode:
                prompt_str += " [bold red]([SHIELD] SAFE)[/bold red]"
            prompt_str += " > "

            user_input = Prompt.ask(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended. Goodbye![/yellow]")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ["exit", "/exit", "/quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if lower == "/help":
            console.print(HELP_TEXT)
            continue

        if lower == "/tools":
            print_tools_table()
            continue

        if lower == "/safe":
            agent.safe_mode = not agent.safe_mode
            status = "[bold green]ENABLED (Will prompt for commands/edits)[/bold green]" if agent.safe_mode else "[bold yellow]DISABLED (Autonomous mode)[/bold yellow]"
            console.print(f"Safe Mode is now {status}")
            continue

        if lower == "/clear":
            agent.reset()
            console.print("[green]Memory cleared! Started fresh conversation.[/green]")
            continue

        if lower.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                console.print(f"Current Model: [bold green]{agent.model}[/bold green]")
            else:
                agent.model = parts[1].strip()
                console.print(f"Switched model to: [bold green]{agent.model}[/bold green]")
            continue

        # Execute Agent Loop with Rich Live Feedback
        try:
            with console.status("[bold cyan]Agent is thinking & planning...[/bold cyan]", spinner="dots") as status_indicator:
                generator = agent.step_stream(user_input)
                final_answer = None

                for event in generator:
                    if event.type == "thought":
                        status_indicator.update(f"[bold cyan]Agent Thought:[/bold cyan] {event.data['content'][:80]}...")

                    elif event.type == "tool_start":
                        t_name = event.data["tool_name"]
                        t_args = event.data["arguments"]
                        status_indicator.stop()

                        arg_str = json.dumps(t_args, indent=2)
                        console.print(Panel(
                            f"[bold yellow]Tool Call:[/bold yellow] [bold green]{t_name}[/bold green]\n"
                            f"[dim]Arguments:[/dim]\n{arg_str}",
                            border_style="yellow",
                            title="⚡ Agent Action",
                            subtitle=f"Step",
                        ))
                        status_indicator.start()
                        status_indicator.update(f"[bold yellow]Executing {t_name}...[/bold yellow]")

                    elif event.type == "tool_end":
                        t_name = event.data["tool_name"]
                        result = event.data["result"]
                        status_indicator.stop()

                        # Truncate preview if very long
                        preview = result if len(result) <= 600 else result[:600] + f"\n... [{len(result) - 600} characters truncated]"
                        console.print(Panel(
                            preview,
                            border_style="blue",
                            title=f"📥 Result from {t_name}",
                        ))
                        status_indicator.start()
                        status_indicator.update("[bold cyan]Analyzing results & determining next step...[/bold cyan]")

                    elif event.type == "tool_denied":
                        status_indicator.stop()
                        console.print(f"[bold red]⛔ Denied:[/bold red] Tool '{event.data['tool_name']}' was blocked by user.")
                        status_indicator.start()

                    elif event.type == "error":
                        status_indicator.stop()
                        console.print(f"[bold red]Error:[/bold red] {event.data['message']}")
                        status_indicator.start()

                    elif event.type == "final_answer":
                        final_answer = event.data["content"]

            # Render final answer formatted as Markdown
            if final_answer:
                console.print(Panel(
                    Markdown(final_answer),
                    border_style="green",
                    title="🤖 Agent Response",
                    subtitle="Finished",
                ))
        except Exception as e:
            console.print(f"[bold red]Execution error:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
