"""
OpenRouter LLM Client Module
Provides easy access to OpenRouter models using OpenAI-compatible SDK.
"""

import os
from typing import Generator, List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env file
load_dotenv(override=True)

# Helper to read environment variable with fallback options
def _get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    if val:
        return val.strip()
    # Fallback to parse manually if dotenv had formatting quirks
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k_clean = k.strip().replace(" ", "_").upper()
                    target_clean = key.strip().replace(" ", "_").upper()
                    if k_clean == target_clean or k.strip().lower() == key.strip().lower():
                        return v.strip()
    return default

OPENROUTER_API_KEY = _get_env("OPENROUTER_API_KEY") or _get_env("opnerouter api key")
DEFAULT_MODEL = _get_env("OPENROUTER_MODEL") or _get_env("MODEL") or "inclusionai/ling-3.0-flash-vl:free"
SITE_URL = _get_env("SITE_URL", "http://localhost:8501")
SITE_NAME = _get_env("SITE_NAME", "AI-App")

def get_client() -> OpenAI:
    """Initialize and return an OpenAI client configured for OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OpenRouter API key is missing! Please set OPENROUTER_API_KEY in your .env file."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
        },
    )

def ask(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    stream: bool = False,
    temperature: float = 0.7,
    **kwargs: Any,
) -> Any:
    """
    Send a single prompt to the OpenRouter model.

    Args:
        prompt: The user query or instruction.
        system_prompt: Optional system instruction.
        model: Model name (defaults to configured model from .env).
        stream: If True, yields streaming chunks; if False, returns complete string.
        temperature: Sampling temperature (default: 0.7).
        **kwargs: Additional parameters passed to client.chat.completions.create.

    Returns:
        String response if stream=False, or a Generator yielding string tokens if stream=True.
    """
    client = get_client()
    model_name = model or DEFAULT_MODEL
    # Some OpenRouter reasoning models can consume the provider's default
    # completion budget before emitting visible answer text.
    kwargs.setdefault("max_tokens", 2048)

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if stream:
        def _generator() -> Generator[str, None, None]:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                temperature=temperature,
                **kwargs,
            )
            for chunk in response:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content

        return _generator()
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""


class ChatSession:
    """
    Maintains a multi-turn conversation history with the model.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def send(self, message: str, stream: bool = False, **kwargs: Any) -> Any:
        """
        Send a message in the conversation.

        Args:
            message: User input message.
            stream: If True, yields streaming chunks and records the full assistant reply.
                    If False, returns full assistant response string.
        """
        client = get_client()
        self.history.append({"role": "user", "content": message})
        kwargs.setdefault("max_tokens", 2048)

        if stream:
            def _stream_gen() -> Generator[str, None, None]:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    stream=True,
                    temperature=self.temperature,
                    **kwargs,
                )
                collected: List[str] = []
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        collected.append(delta.content)
                        yield delta.content
                full_reply = "".join(collected)
                self.history.append({"role": "assistant", "content": full_reply})

            return _stream_gen()
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=self.history,
                stream=False,
                temperature=self.temperature,
                **kwargs,
            )
            reply = response.choices[0].message.content or ""
            self.history.append({"role": "assistant", "content": reply})
            return reply

    def clear(self) -> None:
        """Reset conversation history."""
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    def get_history(self) -> List[Dict[str, str]]:
        """Return copy of the current message history."""
        return list(self.history)
