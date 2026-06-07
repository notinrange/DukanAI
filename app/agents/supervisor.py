"""
Gemini Flash classifies every inbound messages into categories  are 
product | order | lead | booking | human
and detects language: en | hi | hinglish
"""

from langchain_core.messages import HumanMessage

from app.state import DukanState
from app.config import GEMINI_CHAT_MODEL, LLM_TEMPERATURE_CLASSIFY
from app.llm import invoke_gemini_with_fallback
from utils.message_utils import extract_text_content

# prompt

SUPERVISOR_PROMPT = """
        You are a message classifier for an Indian small business WhatsApp AI assistant.
        
        Classify the customer message into EXACTLY ONE intent and detect language.
        Respond in this EXACT three-line format (no other text):
        
        INTENT: <intent>
        LANGUAGE: <language>
        CONFIDENCE: <float 0.0-1.0>
        
        ─── Intent options ───────────────────────────────────────────────────────────
        product   – asking about products, items, prices, availability, catalog,
                    "kya hai", "kitna ka", "stock mein hai", "do you have"
        order     – asking about an EXISTING order status, delivery, return, refund,
                    "order kahan hai", "track my order", "return karna hai"
        lead      – asking about services, courses, coaching, fees, how to join,
                    demo requests, "batao fees kya hai", "how do I enroll"
        booking   – requesting an appointment, slot, consultation, demo class,
                    "booking karni hai", "Saturday slot available?", "appointment lena hai"
        human     – angry/frustrated customer, highly complex multi-part question,
                    urgent issue, or intent genuinely unclear after reading twice
        
        ─── Language options ─────────────────────────────────────────────────────────
        en        – English only
        hi        – Hindi (Devanagari script OR full romanised Hindi)
        hinglish  – Mix of Hindi words and English words in same sentence
        
        ─── Message to classify ──────────────────────────────────────────────────────
        {message}
        """

# Fallback heuristics (used if LLM calls fails)

_HINDI_KEYWORDS= {
      "kya", "hai", "ka", "ki", "ke", "nahi", "haan", "chahiye", "kitna","kahan", "bhai", "yaar", "bata", "dena", "lena", "mujhe", "aap",
    "price", "paisa", "rupee", "order", "booking", "appointment",
}

_PRODUCT_KEYWORDS = {"price", "kitna", "cost", "rate", "available", "stock",
                     "product", "item", "color", "size", "kya hai"}
_ORDER_KEYWORDS = {"order", "delivery", "track", "return", "refund", "status",
                   "arrive", "kahan hai", "aaya nahi"}
_LEAD_KEYWORDS = {
    "fees", "enroll", "join", "course", "coaching", "service",
    "demo", "how to", "kaise", "batao", "information",
    "know more", "more about", "tell me", "details",
    "interested", "inquiry", "enquiry", "services",
    "my name", "name is", "contact", "number is",    # ← add these
    "wholesale", "bulk", "order", "price list",       # ← add these
}
_BOOKING_KEYWORDS = {"appointment", "book", "slot", "available", "saturday",
                     "sunday", "timing", "schedule", "booking", "milna hai"}


def _detect_language_heuristric(text:str) ->str:
    # if any hindi character occurs
    hindi_script_chars = sum(1 for c in text if "\u0900" <= c <="\u097F")

    if hindi_script_chars > 1:
        return "hi"
    words = set(text.lower().split())
    hindi_hit = len(words & _HINDI_KEYWORDS)
    if hindi_hit >= 2:
        return "hinglish"
    if hindi_hit == 1:
        return "hinglish"

    return "en"


def _classify_intent_heuristic(text:str)->tuple[str,float]:
    words = set(text.lower().split())
    if words & _BOOKING_KEYWORDS:
        return "booking",0.5
    if words & _ORDER_KEYWORDS:
        return "order", 0.5
    if words & _LEAD_KEYWORDS:
        return "lead",0.5
    if words & _PRODUCT_KEYWORDS:
        return "product", 0.5
    return "human", 0.3


def supervisor_node(state: DukanState)->dict:
    """
    Classify intent and detect language from the last customer message.
    Return partial state update - only the fields it owns.
    """

    # Grab last HumanMessage

    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_msg = extract_text_content(msg.content)
            break

    if not last_human_msg:
        return {
            "intent": "human",
            "confidence" : 0.0,
            "language" : "en",
            "requires_human" : True,
        }
    
    # LLM Classification

    intent = "human"
    language = _detect_language_heuristric(last_human_msg)
    confidence = 0.8

    try:
        response = invoke_gemini_with_fallback(
            [HumanMessage(content=SUPERVISOR_PROMPT.format(message=last_human_msg))],
            model=GEMINI_CHAT_MODEL,
            temperature=LLM_TEMPERATURE_CLASSIFY,
            caller="supervisor",
        )

        raw = extract_text_content(response.content).strip()

        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("INTENT:"):
                val = line.split(":",1)[1].strip().lower()
                if val in {"product", "order", "lead", "booking", "human"}:
                    intent = val
            elif line.startswith("LANGUAGE:"):
                val = line.split(":",1)[1].strip().lower()
                if val in {"en", "hi", "hinglish"}:
                    language = val
            elif line.startswith("CONFIDENCE:"):
                try: 
                    confidence = float(line.split(":",1)[1].strip())
                except ValueError:
                    pass

    except Exception as exc:
        print(f"[supervisor] LLM call failed ({exc}), using heuristic fallback")
        intent, confidence = _classify_intent_heuristic(last_human_msg)
        language = _detect_language_heuristric(last_human_msg)

    return {
        "intent": intent,
        "confidence" : confidence,
        "language" : language,
        "required_human":intent == "human"
    }


def route_by_intent(state:DukanState)->str:
    """Return the name of the next node based on classified intent"""
    return state.get("intent", "human")

