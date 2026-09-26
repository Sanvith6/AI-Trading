"""
Streamlit Web Chat UI for OpenRouter
Run with: streamlit run app.py
"""

import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Helper to retrieve config
def get_env_var(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val.strip()
    return default

st.set_page_config(
    page_title="AI Chat - OpenRouter",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    env_api_key = get_env_var("OPENROUTER_API_KEY")
    api_key_input = st.text_input(
        "OpenRouter API Key",
        value=env_api_key,
        type="password",
        help="Reads from .env by default. You can override it here.",
    )

    env_model = get_env_var("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-vl:free")
    model_name = st.text_input(
        "Model Name",
        value=env_model,
        help="OpenRouter model ID (e.g. inclusionai/ling-3.0-flash-vl:free, deepseek/deepseek-r1:free, meta-llama/llama-3.3-70b-instruct:free)",
    )

    system_prompt = st.text_area(
        "System Prompt (Instructions)",
        value="You are a helpful, brilliant, and concise AI assistant.",
        rows=3,
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        **Status:**  
        🟢 Connected to OpenRouter  
        ⚡ Streaming enabled
        """
    )

# App Header
st.markdown('<div class="main-title">✨ OpenRouter AI Playground</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">Model: <span class="badge">{model_name}</span></div>',
    unsafe_allow_html=True,
)

# Initialize Chat Messages in Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Type your message here..."):
    if not api_key_input:
        st.error("Please provide an OpenRouter API key in the sidebar or .env file.")
        st.stop()

    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare message payload with system instruction
    messages_payload = []
    if system_prompt.strip():
        messages_payload.append({"role": "system", "content": system_prompt.strip()})
    for m in st.session_state.messages:
        messages_payload.append({"role": m["role"], "content": m["content"]})

    # Call OpenRouter with streaming
    with st.chat_message("assistant"):
        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key_input,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "OpenRouter-Streamlit-App",
                },
            )

            stream = client.chat.completions.create(
                model=model_name,
                messages=messages_payload,
                stream=True,
                temperature=temperature,
            )

            def stream_generator():
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yield delta.content

            response_content = st.write_stream(stream_generator())
            st.session_state.messages.append({"role": "assistant", "content": response_content})

        except Exception as e:
            st.error(f"Error calling OpenRouter: {e}")
