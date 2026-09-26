"""
AI Agent Web Dashboard (Streamlit)
Visual interface for interacting with the autonomous agent, inspecting tool calls, and browsing files.
Run with: streamlit run agent_app.py
"""

import os
import json
import streamlit as st
from dotenv import load_dotenv

from agent import Agent, AgentEvent
import tools
from llm_client import DEFAULT_MODEL

load_dotenv(override=True)

st.set_page_config(
    page_title="AI Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern theme
st.markdown(
    """
    <style>
    .agent-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .agent-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .tool-pill {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        padding: 0.2rem 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: monospace;
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {"role": "user"|"assistant", "content": str, "tool_runs": list}

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Agent Configuration")

    model = st.text_input(
        "Model ID",
        value=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        help="OpenRouter model name",
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.4, step=0.1)
    max_steps = st.slider("Max Reasoning Steps", min_value=3, max_value=25, value=12, step=1)

    st.markdown("---")
    st.markdown("### 🛠️ Available Agent Tools")
    st.markdown(
        """
        - 💻 `execute_command` (PowerShell)
        - 📄 `read_file`
        - ✍️ `write_file`
        - 📁 `list_directory`
        - 🔍 `search_files`
        - 🐍 `run_python` (Isolated subprocess)
        - 🌐 `fetch_url` (Web / REST)
        """
    )

    st.markdown("---")
    st.markdown("### 📁 Workspace Files")
    try:
        files = [f for f in os.listdir(tools.WORKSPACE_DIR) if not f.startswith((".", "__"))]
        selected_file = st.selectbox("Inspect a file:", ["-- Select --"] + sorted(files))
        if selected_file and selected_file != "-- Select --":
            with st.expander(f"View `{selected_file}`", expanded=True):
                content = tools.read_file(selected_file)
                st.code(content, language="python" if selected_file.endswith(".py") else "text")
    except Exception as e:
        st.caption(f"Could not list files: {e}")

    st.markdown("---")
    if st.button("🗑️ Clear Conversation Memory", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Main UI Header
st.markdown('<div class="agent-header">🤖 Autonomous AI Agent</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="agent-sub">Powered by <b>{model}</b> on OpenRouter • Full System & Workspace Tool Access</div>',
    unsafe_allow_html=True,
)

# Suggested Quick Task Prompts
cols = st.columns(4)
quick_task = None
with cols[0]:
    if st.button("📋 List Workspace Files", use_container_width=True):
        quick_task = "List all files in the current workspace directory and explain what each is for."
with cols[1]:
    if st.button("🐍 Test Python Math", use_container_width=True):
        quick_task = "Use run_python to calculate the first 10 Fibonacci numbers and print them."
with cols[2]:
    if st.button("🔍 Search Project Code", use_container_width=True):
        quick_task = "Search for where 'OPENROUTER_API_KEY' is referenced across all files."
with cols[3]:
    if st.button("🌐 Fetch HackerNews Top", use_container_width=True):
        quick_task = "Fetch 'https://news.ycombinator.com' and summarize the top 3 headline topics."

# Display Previous Conversation History
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        # Show tool runs in accordion if any
        if turn.get("tool_runs"):
            with st.expander(f"🛠️ Tool Invocations ({len(turn['tool_runs'])})", expanded=False):
                for t in turn["tool_runs"]:
                    st.markdown(f"**Tool:** `{t['name']}`")
                    st.json(t["args"])
                    st.caption("Result:")
                    st.code(t["result"], language="text")
                    st.divider()
        st.markdown(turn["content"])

# Chat Input Handler
user_prompt = st.chat_input("Give your agent a task (e.g., 'Inspect the project and create a new script')...")
prompt_to_run = quick_task or user_prompt

if prompt_to_run:
    # Append user turn
    st.session_state.chat_history.append({"role": "user", "content": prompt_to_run, "tool_runs": []})
    with st.chat_message("user"):
        st.markdown(prompt_to_run)

    # Initialize Agent
    agent = Agent(
        model=model,
        max_steps=max_steps,
        temperature=temperature,
    )

    # Restore prior conversation context if exists
    for turn in st.session_state.chat_history[:-1]:
        if turn["role"] in ["user", "assistant"]:
            agent.messages.append({"role": turn["role"], "content": turn["content"]})

    # Run Agent Loop with Live UI Updates
    with st.chat_message("assistant"):
        status_box = st.status("Agent initialized and reasoning...", expanded=True)
        tool_runs = []
        final_answer = ""

        try:
            for event in agent.step_stream(prompt_to_run):
                if event.type == "step_start":
                    status_box.update(label=f"Step {event.data['step']} / {event.data['max_steps']}...", state="running")

                elif event.type == "thought":
                    status_box.write(f"💭 **Thought:** {event.data['content']}")

                elif event.type == "tool_start":
                    status_box.write(f"⚡ **Calling tool:** `{event.data['tool_name']}`")
                    status_box.json(event.data["arguments"])

                elif event.type == "tool_end":
                    status_box.write(f"✅ **Completed:** `{event.data['tool_name']}`")
                    tool_runs.append({
                        "name": event.data["tool_name"],
                        "args": event.data.get("arguments", {}),
                        "result": event.data["result"],
                    })

                elif event.type == "final_answer":
                    final_answer = event.data["content"]
                    status_box.update(label="Task Completed!", state="complete", expanded=False)
                    st.markdown(final_answer)

                elif event.type == "error":
                    status_box.update(label=f"Error: {event.data['message']}", state="error")
                    st.error(event.data["message"])

            # Save assistant message in session history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_answer,
                "tool_runs": tool_runs,
            })

        except Exception as e:
            st.error(f"Unexpected execution error: {str(e)}")
