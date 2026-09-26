"""
Quickstart Example
Run this script to send a single query to your configured OpenRouter model.
"""

from llm_client import ask, DEFAULT_MODEL

def main():
    print("=" * 60)
    print(f"Using Model: {DEFAULT_MODEL}")
    print("=" * 60)

    prompt = "Explain quantum computing in 2 simple sentences."
    print(f"\nUser: {prompt}\n")

    print("AI (Streaming): ", end="", flush=True)
    for token in ask(prompt, stream=True):
        print(token, end="", flush=True)
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
