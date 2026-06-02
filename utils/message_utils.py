from typing import Any


def extract_text_content(content: Any) -> str:
    """
    Safely extract plain text from LangChain message content.

    Handles:
    - str
    - list[str]
    - list[dict]
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts).strip()

    return ""

def _extract_last_ai_message(state: dict) -> str:
    """Return the content of the last AIMessage in state, or empty string."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai":
            return msg.content
    return ""