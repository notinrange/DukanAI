"""Uses catalog_retriever @tool (pgvector cosine search) to answer
"price kya hai?", "do you have red in size M?", etc.
Responds in the customer's detected language.
"""

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from app.state import DukanState
from app.rag.catalog_rag import make_catalog_retriever_tool
from app.config import GEMINI_CHAT_MODEL, LLM_TEMPERATURE_CHAT
from app.llm import invoke_gemini_with_fallback
from utils.message_utils import extract_text_content

_SYSTEM_PROMPT = """\
    You are a helpful product assistant for an Indian small business.
    
    Rules:
    - Answer ONLY about products in the catalog context provided below.
    - Use ₹ for all Indian Rupee prices.
    - Keep the reply short — max 200 words, suitable for WhatsApp.
    - Do NOT use markdown headers (##). You may use *bold* for product names.
    - If the product is not in the catalog, say so honestly.
    - Respond in {language}:
        en        → English
        hi        → Hindi (Devanagari preferred)
        hinglish  → Friendly mix of Hindi + English
    
    Catalog context:
    {catalog_ctx}
    """

_USER_PROMPT = """\
    Customer question: {question}
 
    Answer the question using only the catalog context in the system prompt.
    """

def product_agent_node(state: DukanState)->dict:
    """
    Handle product queries using pgvector RAG.
    Writes reply to messages and sets agent_used, catalog_ctx.
    """
    business_id = state["business_id"]
    language  = state.get("language","en")

    # get last customer message
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            last_human_msg = extract_text_content(msg.content)
            break

    # RAG retrieval

    catalog_ctx = ""

    try:
        retriever = make_catalog_retriever_tool(business_id)
        catalog_ctx = retriever.invoke(last_human_msg)
    except Exception as exc:
        print(f"[product_agent] catelog_retriever error: {exc}")
        catalog_ctx = "Catalog temporarily unavailable"


    # llm answer
    try:
        message = [
            SystemMessage(
                content=_SYSTEM_PROMPT.format(language=language, catalog_ctx = catalog_ctx)
            ),
            HumanMessage(content=_USER_PROMPT.format(question=last_human_msg))
        ]

        response = invoke_gemini_with_fallback(
            message,
            model=GEMINI_CHAT_MODEL,
            temperature=LLM_TEMPERATURE_CHAT,
            caller="product_agent",
        )
        reply = extract_text_content(response.content).strip()

    except Exception as exc:
        print(f"[product_agent] LLM error: {exc}")
        reply = (
            "मुझे अभी catalog access करने में दिक्कत हो रही है। थोड़ी देर में try करें।"
            if language in ("hi", "hinglish")
            else "I'm having trouble accessing the product catalog right now. Please try again shortly!"
        )
    
    return {
        "messages" : [AIMessage(content=reply)],
        "catalog_ctx" : catalog_ctx,
        "agent_used" : "product_agent"
    }
