"""
Final node before END.
- Trims length to WhatsApp-safe MAX_REPLY_CHARS
- Strip markdown headers that don't render on WhatsApp
- Remove multiple black lines.
- Ensures messages end with a single trailing newline for readability.
"""
import re

from langchain_core.messages import AIMessage
from app.state import DukanState
from app.config import MAX_REPLY_CHARS

from utils.message_utils import extract_text_content
def response_formatter(state: DukanState)->dict:
    """
    Format the last AI message for Whatsapp delivery.
    Return empty dict if not AIMessage found (no-op).
    """

    # Find the last AI Message
    messages = list(state.get("messages",[]))
    if not messages:
        return {}
    
    last_ai_idx = None
    for i in range(len(messages)-1, -1,-1):
        if messages[i].type == "ai":
            last_ai_idx = i
            break
    
    if last_ai_idx is None:
        return {}
    
    text: str = extract_text_content(messages[last_ai_idx].content)

    # Formatting passes 

    # 1 strip markdown headers
    text = re.sub(r"^#{1,6}\s+","", text, flags = re.MULTILINE)

    # 2 Collapse 3+ blank lines -> 1 blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3 Strip leading and trailing whitespace
    text = text.strip()

    # 4 Hard truncate at MAX_REPLY_CHARTS (keep last complete sentence if possible)
    if len(text) > MAX_REPLY_CHARS:
        truncated = text[:MAX_REPLY_CHARS - 3]
        # Try to cut at last sentence boundary
        last_period = max(
            truncated.rfind("|"),
            truncated.rfind(", "),
            truncated.rfind("?\n"),
            truncated.rfind("!\n"),
        )
        if last_period > MAX_REPLY_CHARS //2:
            text = truncated[: last_period + 1].strip() + " ..."
        else:
            text = truncated.strip() + "..."
    
    # If the text hasn't changed, skip the message list update
    if text == extract_text_content(messages[last_ai_idx].content):
        return {}
    
    # Replace the last AI Message with the formatted version
    new_messages = messages[:last_ai_idx] + [
        AIMessage(content=text)
    ] + messages[last_ai_idx + 1:]

    return {"messages": new_messages}
