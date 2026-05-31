class DukanState(TypeDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    business_id: str
    intent: str         # product | order | lead | booking
    confidence: float   # 0.0 - 1.0
    catalog_ctx: Optional[str] # RAG retreived text
    requires_human: bool
    language: str       # en | hi | hinglish
    booking_data: Optional[dict]
    lead_data: Optional[dict]