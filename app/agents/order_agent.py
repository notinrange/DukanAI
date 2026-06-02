"""
Order Status Agent
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

from app.state import DukanState
from app.config import GOOGLE_API_KEY, GEMINI_CHAT_MODEL, LLM_TEMPERATURE_CHAT

from utils.message_utils import extract_text_content
_llm = ChatGoogleGenerativeAI(
    model = GEMINI_CHAT_MODEL,
    google_api_key = GOOGLE_API_KEY,
    temperature= LLM_TEMPERATURE_CHAT,
)


_ORDER_PROMPT = """\
You are an order support assistant for an Indian small business WhatsApp agent.
 
The customer is asking about an order. Ask them to share their order number so you
can look it up. Be friendly and reassuring.
 
Respond in {language}:
  en        → English
  hi        → Hindi
  hinglish  → Hindi + English mix
 
Customer message: {message}
Keep reply under 80 words, suitable for WhatsApp.
"""


def order_agent_node(state:DukanState) -> dict:
    """
    Order status agent.
    Currently asks for order number - DB lookup added in Week 3.
    """

    language = state.get("language","en")

    last_human_msg = ""

    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_msg = extract_text_content(msg.content)
            break

    try:
        response = _llm.invoke([
            HumanMessage(content= _ORDER_PROMPT.format(
                language = language,
                message = last_human_msg
            ))
        ])

        reply = extract_text_content(response.content).strip()
    except Exception as exc:
        print(f"[order_agent] LLM error: {exc}")
        if language in ("hi","hinglish"):
            reply = "Apna order number share karein, mein abhi check karta hoon! 📦"
        else:
            reply = "Please share your order number and I'll check the status for you right away!📦"

    return {
        "messages" : [AIMessage(content=reply)],
        "agent_used" : "order_agent"
    }

