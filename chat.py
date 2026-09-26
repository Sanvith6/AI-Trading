"""
Interactive Terminal Chat with OpenRouter
Run this with: python chat.py
"""

import sys
from llm_client import ChatSession, DEFAULT_MODEL

BANNER = rf"""
============================================================
           OPENROUTER INTERACTIVE TERMINAL CHAT
============================================================
 Active Model : {DEFAULT_MODEL}
 Commands     :
   /clear          - Clear conversation history
   /model [name]   - View or switch active model
   /system [prompt]- Set custom system instructions
   /help           - Display this help banner
   /exit or exit   - End chat session
============================================================
"""

def main():
    print(BANNER)
    session = ChatSession()

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ["exit", "/exit", "/quit"]:
            print("Goodbye!")
            break

        if lower == "/clear":
            session.clear()
            print("[History cleared! Starting fresh conversation.]")
            continue

        if lower.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                print(f"[Current model: {session.model}]")
            else:
                new_model = parts[1].strip()
                session.model = new_model
                print(f"[Switched model to: {new_model}]")
            continue

        if lower.startswith("/system"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                print(f"[Current system prompt: {session.system_prompt or 'None'}]")
            else:
                new_sys = parts[1].strip()
                session.system_prompt = new_sys
                session.clear()
                print(f"[System prompt set & history reset: {new_sys}]")
            continue

        if lower == "/help":
            print(BANNER)
            continue

        print(f"\n{session.model.split('/')[-1]} > ", end="", flush=True)
        try:
            stream = session.send(user_input, stream=True)
            for token in stream:
                print(token, end="", flush=True)
            print()
        except Exception as e:
            print(f"\n[Error communicating with OpenRouter: {e}]")

if __name__ == "__main__":
    main()
