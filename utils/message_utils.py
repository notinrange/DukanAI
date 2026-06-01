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