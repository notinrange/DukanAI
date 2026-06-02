"""
 
Key design: This node is compiled with interrupt_before=["human_handoff"].
That means the graph PAUSES before this node ever executes.
 
Normal flow (escalation):
  1. supervisor classifies intent = "human"
  2. graph routes to human_handoff
  3. LangGraph interrupt_before fires → graph PAUSES, saves state to Redis
  4. dukan-brain returns {requires_human: true} to dukan-api
  5. dukan-api sends a template "owner will contact you" message to customer
  6. Merchant is notified on dashboard (red badge)
  7. Merchant replies via /agent/resume → graph RESUMES from saved state
  8. THIS NODE now runs — it inserts a bridging message and control
     returns to AI for subsequent messages
 
During human state:
  - dukan-api blocks any AI reply if conversation.status == "human"
  - Only merchant messages are forwarded to the customer
"""

from langchain_core.messages import AIMessage
from app.state import DukanState

_BRIDGE_MESSAGES = {
    "en": (
        "The business owner has responded above. "
        "I'm back to assist you — feel free to continue! 😊"
    ),
    "hi": (
        "Owner sahab ne aapko reply kar diya hai. "
        "Main phir se available hoon — kuch aur chahiye toh batayein! 😊"
    ),
    "hinglish": (
        "Owner ne reply kar diya! Main phir se help ke liye available hoon. "
        "Kuch aur chahiye? 😊"
    ),
}


def human_handoff_node(state: DukanState)->dict:
    """
    Human handoff node.

    This runs only when the graph RESUMES after a merchant reply via /agent/resume/. It adds a bridging message so the customer knows the AI is available again.

    During a normal escalation, interrupt_before fires BEFORE this node and the graph never reaches here until the merchant resumes it.

    """

    langugage = state.get("language", "en")
    bridge = _BRIDGE_MESSAGES.get(langugage, _BRIDGE_MESSAGES["en"])

    return {
        "messages" : [AIMessage(content=bridge)],
        "requires_human" : False,
        "agent_used" : "human_handoff"
    }
