"""
Multi - turn flow : captures service ->date ->time from conversation.
State persists across turns via Redis checkpointer.
Saves confirmed booking to PostgreSQL booking table
"""

import json
from langchain_core.messages import AIMessage, HumanMessage

from app.state import DukanState
from app.config import GEMINI_CHAT_MODEL, LLM_TEMPERATURE_CHAT, DATABASE_URL
from app.llm import invoke_gemini_with_fallback

from utils.message_utils import extract_text_content

# Prompts

_EXTRACT_PROMPT = """\
Extract appointment booking information from this WhatsApp conversation.
Return ONLY a valid JSON object (null for missing fields):
{{
  "service": "what service/appointment they want (e.g. 'haircut', 'demo class', 'consultation')",
  "date": "YYYY-MM-DD if a specific date is mentioned, else null",
  "time": "HH:MM in 24h format if a specific time is mentioned, else null",
  "customer_name": "customer name if mentioned, else null"
}}
 
Conversation:
{conversation}
 
Return ONLY the JSON object. No markdown, no explanation.
"""

_RESPOND_PROMPT = """\
You are a friendly booking assistant for an Indian small business on WhatsApp.
Help the customer book an appointment in a natural, conversational way.
 
Booking info collected so far:
{booking_data}
 
Missing fields: {missing}
 
Recent customer message: "{message}"
 
Instructions:
- service missing: Ask what they'd like to book (haircut, demo class, consultation, etc.)
- date missing: Ask for their preferred date (mention a few available days if unsure)
- time missing: Ask for preferred time (morning / afternoon / evening is fine to offer)
- all collected: Confirm the booking with full details and say they'll receive a confirmation
- Language: respond in {language} (en=English, hi=Hindi, hinglish=Hindi+English mix)
- Friendly, conversational, max 120 words, WhatsApp-appropriate
"""

# DB persistance

def _save_booking(state: DukanState, booking_data:dict) -> None:
    """Save booking to PostgreSQL. Slently skips if DB unavailable"""

    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bookings
                    (business_id, customer_phone, customer_name, service,
                    booking_date, booking_time, status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                """,
                (
                    state["business_id"],
                    state.get("customer_id", ""),
                    booking_data.get("customer_name"),
                    booking_data.get("service"),
                    booking_data.get("date"),     # None → NULL in PG (fine)
                    booking_data.get("time"),
                ),
            )
            conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[booking_agent] DB save skipped: {exc}")


# Node

def booking_agent_node(state:DukanState)->dict:
    """
    Multi-turn appointment booking.
    - Extracts service / date / time from conversation
    - Merges into persisted booking_data (via Redis checkpointer)
    - Asks for next missing field
    - Saves to bookings table once service + date + time are captured
    """
    language = state.get("language","en")
    booking_data:dict = dict(state.get("booking_data")or {})

    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_msg = extract_text_content(msg.content)
            break

    # Extract booking fields

    recent_msgs = state["messages"][-6:]
    conversation_text = "\n".join(
        f"{'Customer' if m.type == 'human' else 'Agent'}: {m.content}"
        for m in recent_msgs
    )

    try:
        extraction = invoke_gemini_with_fallback([
            HumanMessage(content = _EXTRACT_PROMPT.format(conversation=conversation_text))
        ], model=GEMINI_CHAT_MODEL, temperature=LLM_TEMPERATURE_CHAT, caller="booking_agent")
        raw = extract_text_content(extraction.content).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted: dict = json.loads(raw)

        for key in ("service","date","time","customer_name"):
            if extracted.get(key) and not booking_data.get(key):
                booking_data[key] = extracted[key]
    except Exception as exc:
        print(f"[booking_agent] extraction error: {exc}")

    # Determine what's still missing  
    
    # customer_name is optional — we only require service + date + time
    missing = [k for k in ("service", "date", "time") if not booking_data.get(k)]

    # Save if Complete
    if not missing:
        _save_booking(state, booking_data)
    
    # Generate response
    try:
        response = invoke_gemini_with_fallback([
            HumanMessage(content=_RESPOND_PROMPT.format(
                booking_data = json.dumps(booking_data, ensure_ascii = False, indent = 2),
                missing=", ".join(missing) if missing else "none - booking confirmed",
                message = last_human_msg,
                language = language,
            ))
        ], model=GEMINI_CHAT_MODEL, temperature=LLM_TEMPERATURE_CHAT, caller="booking_agent")

        reply = extract_text_content(response.content).strip()
    except Exception as exc:
        print(f"[booking_agent] respond error: {exc}")
        reply = (
            "Appomintment book karne ke liye kaun si service chaiye aur kab?"
            if language in ("hi","hinglish")
            else "I'd love to help you book! What service would you like and when?"
        )
    
    return {
        "messages" : [AIMessage(content=reply)],
        "booking_data": booking_data,
        "agent_used" : "booking_agent",
    }


            
