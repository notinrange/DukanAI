"""
Endpoints (from TRD §4):
  POST /agent/process   — process inbound customer WhatsApp message
  POST /agent/resume    — resume after merchant takeover
  POST /catalog/embed   — embed merchant's product catalog into pgvector
  GET  /health          — uptime check
 
Called exclusively by dukan-api (Spring Boot) over Docker internal network.
Never exposed to the public internet.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn 
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from state import DukanState
from utils.message_utils import _extract_last_ai_message
from app.graph import dukan_graph, make_config

# Life span
@asynccontextmanager
async def lifespan(app:FastAPI):
    print("🤖  dukan-brain starting — LangGraph graph compiled")
    yield
    print("🔴  dukan-brain shutting down")

# app
app = FastAPI(
    title="dukan-brain",
    description="DukanAI Langgraph multi-agent AI service",
    version = "1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url = None,
)

# request/response models

class ProcessRequest(BaseModel):
    message: str = Field(..., description="Raw customer WhatsApp message text.")
    customer_id: str = Field(...,description="Customer's Whatsapp phone.")
    businees_id: str = Field(..., description="Merchant's business UUID.")
    customer_name: Optional[str] = Field(None, description="Customer display name if known.") 

class ProcessResponse(BaseModel):
    message:str
    intent:str
    agent_used : str
    language:str
    requires_human:bool
    latency_ms:int


class ResumeRequest(BaseModel):
    customer_id:str
    business_id:str
    merchant_message: str = Field(..., description="Message the merchant typed to the customer")

class ResumeResponse(BaseModel):
    message: str
    latency_ms:int

class EmbedRequest(BaseModel):
    business_id:str
    s3_key:str = Field(..., description="S3 onject key for the uploaded catalog file")

class EmbedResponse(BaseModel):
    chunks_embedded:int
    status:str

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    print(f"[main] Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500,content={"detail":str(exc)})

# Health

@app.get("/health")
async def agent_process(req: ProcessRequest):
    """
    Main entry point.
    Called by dukan-api on every inbound customer WhatsApp message.
 
    Flow:
      1. Build LangGraph input state (new HumanMessage)
      2. Invoke graph with thread_id = business_id:customer_id
         (Redis checkpointer loads prior conversation history automatically)
      3. Extract AI reply from final state
      4. Return {message, intent, agent_used, requires_human, ...}
 
    If requires_human=True the graph is paused (interrupt_before fired).
    dukan-api sends a template message to the customer and notifies the merchant.
    """
    t0 = time.monotonic()
    config = make_config(req.businees_id, req.customer_id)

    initial_input = {
        "messages" : [HumanMessage(content=req.message)],
        "customer_id": req.customer_id,
        "business_id" : req.businees_id,
        "intent" : "",
        "confidence" : 0.0,
        "catalog_ctx": None,
        "requires_human": False,
        "language": "en",
        "booking_data" : None,
        "lead_data" : None,
        "agent_used" : None,
        "draft_response": None,
    }

    try:
        # Langgraph invoke is synchronus; run in thread pool to avoid blocking the event loop
        final_state: dict = await asyncio.to_thread(
            dukan_graph.invoke, initial_input, config
        ) # type:ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Afent graph error: {exc}")
    
    # Extract last AI reply
    reply_text = _extract_last_ai_message(final_state)

    if final_state.get("requires_human") and not reply_text:
        reply_text=""

    latency_ms = int((time.monotonic() - t0) * 1000)

    return ProcessResponse(
        message = reply_text,
        intent = final_state.get("intent","unknown"),
        agent_used=final_state.get("agent_used") or "none",
        language=final_state.get("language","en"),
        requires_human = bool(final_state.get("requires_human", False)),
        latency_ms=latency_ms,
    )


# POST /agent/resume

@app.post("/agent/resume", response_model = ResumeResponse)
async def agent_resume(req: ResumeRequest):
    """
    Resume a paused LangGraph conversation after merchant reply.
    Called by dukan-api when merchant clicks "Send & Resume AI" on the dashboard.
 
    Flow:
      1. Add merchant's reply as an AIMessage labelled [Merchant]
      2. Reset requires_human = False
      3. Graph resumes from saved Redis state through human_handoff node
      4. human_handoff adds a bridging message (AI is back)
      5. Return bridging message to dukan-api for delivery to customer
    """
    t0 = time.monotonic()
    config = make_config(req.business_id, req.customer_id)

    # Inject merchant reply into conversation history and un-pause
    resume_input = {
        "messages":[
            AIMessage(content=f"[Merchant reply]: {req.merchant_message}")
        ],
        "requires_human" : False,
    }

    try:
        final_state:dict = await asyncio.to_thread(
            dukan_graph.invoke, resume_input, config
        ) # type:ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Resume error: {exc}")
    
    reply_text = _extract_last_ai_message(final_state)
    latency_ms = int((time.monotonic() - t0)*1000)

    return ResumeResponse(message=reply_text or "", latency_ms=latency_ms)


# POST /catalog/embed
@app.post("/catalog/embed", response_model = EmbedResponse)
async def catalog_embed(req: EmbedRequest):
    """
    Download a catalog file from S3, embed with Gemini text-embedding-004,
    and store in pgvector (catalog_items table).
    Called by dukan-api after a merchant uploads their CSV/Excel catalog.
    """
    try:
        from app.rag.catalog_rag import embed_catalog_from_s3
        chunks = await asyncio.to_thread(
            embed_catalog_from_s3, req.business_id, req.s3_key
        )
        return EmbedResponse(chunks_embedded=chunks, status="success")
    except Exception as exc:
        raise HTTPException(status_code=500, detail = f"Catalog embed error: {exc}")
    

# entry point of application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )



