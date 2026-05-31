from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain.graph.message import add_messages


class DukanState(TypeDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    business_id: str
    intent: str         # product | order | lead | booking
    confidence: float   # 0.0 - 1.0
    language: str       # en | hi | hinglish

    agent_used: Optional[str]
    catalog_ctx: Optional[str] # RAG retreived text

    requires_human: bool
    
    booking_data: Optional[dict]
    lead_data: Optional[dict]

    draft_response: Optional[str]