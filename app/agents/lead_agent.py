"""
Multi-turn flow: progressively collects name -> phone -> requirement.
State persists across turns via Redis checkpointer (thread_id = customer phone).
Saves captured lead to PostgreSQL leads table.
"""

import json
from langchain_core.messages import AIMessage, HumanMessage

from app.state import DukanState
from app.config import GEMINI_CHAT_MODEL, LLM_TEMPERATURE_CHAT, DATABASE_URL
from app.llm import invoke_gemini_with_fallback

from utils.message_utils import extract_text_content

_EXTRACT_PROMPT = """\
Extract lead information from the WhatsApp conversation.
Return ONLY a valid JSON object with these fields (use null if not found):
{{
    "name": "customer full name or first name",
    "phone": "10-digit Indian mobile number without +91",
    "requirement": "what service or product they are interested in"
}}

Conversation:
{conversation}
 
Return ONLY the JSON object. No markdown, no explanation.
"""

_RESPOND_PROMPT = """\
You are a friendly sales assistant for an Indian small business on WhatsApp.
Your goal is to qualify a lead by naturally gathering their name, phone number,
and what they're interested in — through friendly conversation, not a form.
 
Collected so far:
{lead_data}
 
Missing fields: {missing}
 
Recent customer message: "{message}"
 
Instructions:
- If name is missing: warmly ask for their name. Don't be robotic.
- If name is known but phone is missing: ask for their WhatsApp / contact number.
- If name + phone known but requirement missing: ask what they need help with.
- If ALL three are collected: thank them and say someone will contact them shortly.
- Never repeat a question already asked in the last 2 messages.
- Language: respond in {language} (en=English, hi=Hindi, hinglish=Hindi+English mix).
- Keep it conversational, max 100 words, WhatsApp-friendly.
"""

# DB persistance

def _save_lead(state: DukanState, lead_data: dict) -> None:
    """Persist lead to PostgreSQL. Silently skips if DB unavailable (dev mode)."""
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO leads (business_id, customer_phone, customer_name, requirement, status)
                VALUES (%s, %s, %s, %s, 'new')
                ON CONFLICT DO NOTHING
                """,
                (
                    state["business_id"],
                    state.get("customer_id", ""),
                    lead_data.get("name"),
                    lead_data.get("requirement"),
                ),
            )
            conn.commit()
        conn.close()
    except Exception as exc:
        print(f"[lead_agent] DB save skipped: {exc}")


# Node

def lead_agent_node(state: DukanState)->dict:
    """
    Multi-turn lead qualification.
    - Extracts what the customer has shared so far via LLM
    - Merges into existing lead_data (persisted via Redis checkpointer)
    - Asks for the next missing field naturally
    - Saves to DB once all three fields (name, phone, requirement) are captured
    """
    language = state.get("language","en")
    lead_data: dict = dict(state.get("lead_data") or {})

    # Last customer message
    last_human_message = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_message = extract_text_content(msg.content)
            break

    # Extract from last 6 messages
    recent_msgs = state["messages"][-6:]
    conversation_text = "\n".join(
        f"{'Customer' if m.type == 'human' else 'Agent'}: {m.content}"
        for m in recent_msgs
    )

    try:
        extraction = invoke_gemini_with_fallback([
            HumanMessage(content=_EXTRACT_PROMPT.format(conversation=conversation_text))
        ], model=GEMINI_CHAT_MODEL, temperature=LLM_TEMPERATURE_CHAT, caller="lead_agent")

        raw = extract_text_content(extraction.content).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        extracted: dict = json.loads(raw)

        # Merge - only fill empty slots, never overwrite confirmed data
        for key in ("name","phone","requirement"):
            if extracted.get(key) and not lead_data.get(key):
                lead_data[key] = extracted[key]
    except Exception as exc:
        print(f"[lead_agent] extraction error: {exc}")
    
    # What is still missing ?
    missing = [k for k in ("name","phone","requirement") if not lead_data.get(k)]

    # Save to DB if complete
    if not missing:
        _save_lead(state,lead_data)
    
    # Generate response

    try:
        response = invoke_gemini_with_fallback([
            HumanMessage(content=_RESPOND_PROMPT.format(
                lead_data = json.dumps(lead_data,ensure_ascii=False, indent=2),
                missing=", ".join(missing) if missing else "none - all collected",
                message = last_human_message,
                language = language,
            ))
        ], model=GEMINI_CHAT_MODEL, temperature=LLM_TEMPERATURE_CHAT, caller="lead_agent")
        reply = extract_text_content(response.content).strip()
    except Exception as exc:
        print(f"[lead_agent] respond error: {exc}")
        reply = (
            "Shukriya! Aapka naam aur contact number share karein — main jald reply karunga. 🙏"
            if language in ("hi", "hinglish")
            else "Thank you for your interest! Could you please share your name and contact number? 🙏"
        )
    return {
        "messages" : [AIMessage(content=reply)],
        "lead_data" : lead_data,
        "agent_used" : "lead_agent",
    }

