from fitz.core.llm.chat.plugins.local import LocalChatClient


def main() -> None:
    llm = LocalChatClient()

    out = llm.chat(
        [
            {"role": "system", "content": "You are a test LLM."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ]
    )

    print("\n=== LOCAL CHAT OUTPUT ===")
    print(out)
    print("========================\n")


if __name__ == "__main__":
    main()
