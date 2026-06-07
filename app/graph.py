"""
 
Topology (from TRD):
 
                    ┌─────────────┐
         ┌──────────│  supervisor │──────────┐
         │          └─────────────┘          │
         │ product/order/lead/booking        │ human
         ▼                                   ▼
  [specialist agent]               [human_handoff] ── END
         │                         (interrupt_before)
         ▼
     [formatter]
         │
        END
 
Redis checkpointer: thread_id = "{business_id}:{customer_phone}"
This gives each customer a separate persistent conversation history,
isolated per business.
"""

import os
from langgraph.graph import StateGraph,END

from app.state import DukanState
from app.agents.supervisor import supervisor_node, route_by_intent
from app.agents.product_agent import product_agent_node
from app.agents.order_agent import order_agent_node
from app.agents.lead_agent import lead_agent_node
from app.agents.booking_agent import booking_agent_node
from app.agents.human_handoff import human_handoff_node
from app.agents.formatter import response_formatter
from app.config import REDIS_URL

from contextlib import ExitStack

stack = ExitStack()

def _build_checkpointer():
    """
    Try Redis checkpointer; fall back to in-memory for local dev without Redis.
    """
    # Try the packaged langgraph-checkpoint-redis first
    try:
        from langgraph.checkpoint.redis import RedisSaver
        

        checkpointer = stack.enter_context(
            RedisSaver.from_conn_string(REDIS_URL)
        )
        checkpointer.setup()

        print(f"[graph] Using Redis checkpointer: {REDIS_URL}")

        return checkpointer
    except Exception as e:
        print(f"[graph] Redis checkpointer failed ({e.__class__.__name__}: {str(e)[:80]})")
        print("[graph] ⚠️  Falling back to in-memory checkpointer...")

    # Dev fallback
    from langgraph.checkpoint.memory import MemorySaver
    print("[graph] ℹ️  Using MemorySaver (in-process, non-persistent)")
    return MemorySaver()



# graph builder

def build_graph():
    """
    Assemble and compile the DukanAI StateGraph.
    Called once at startup; result stored as 'dukan_graph' singleton.
    """
    workflow = StateGraph(DukanState)

    # Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("product_agent", product_agent_node)
    workflow.add_node("order_agent", order_agent_node)
    workflow.add_node("lead_agent", lead_agent_node)
    workflow.add_node("booking_agent", booking_agent_node)
    workflow.add_node("human_handoff", human_handoff_node)
    workflow.add_node("formatter", response_formatter)

    # Entry Point
    workflow.set_entry_point("supervisor")

    # Conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_by_intent,
        {
            "product" : "product_agent",
            "order" : "order_agent",
            "lead" : "lead_agent",
            "booking" : "booking_agent",
            "human" : "human_handoff",
        },
    )

    # Adding Edges
    for node_name in {"product_agent", "order_agent", "lead_agent", "booking_agent"}:
        workflow.add_edge(node_name, "formatter")
    workflow.add_edge("formatter", END)

    workflow.add_edge("human_handoff", END)

    checkpointer = _build_checkpointer()

    compiled = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_handoff"],
    )

    return compiled


# Singleton
dukan_graph = build_graph()

def make_config(business_id: str, customer_id:str)->dict:
    """
    Build the LangGraph config dict for a specific conversation.
    thread_id ensures each customer↔business pair has isolated history.
    """
    return {
        "configurable":{
            "thread_id" : f"{business_id}:{customer_id}"
        }
    }