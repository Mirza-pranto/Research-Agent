import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import workflow, get_research_graph

research_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_graph
    async with AsyncSqliteSaver.from_conn_string("research_memory.db") as checkpointer:
        research_graph = get_research_graph(checkpointer)
        print("--- [DATABASE] AsyncSqliteSaver Checkpointer active ---")
        yield

app = FastAPI(
    title="Free AI Research Agent API",
    description="Backend API powering the LangGraph multi-agent research workflow.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: Optional[str] = None
    query: Optional[str] = None
    thread_id: Optional[str] = None
    auto_approve: bool = True

    def get_search_topic(self) -> str:
        search_term = self.topic or self.query
        if not search_term or not search_term.strip():
            raise ValueError("Research topic/query cannot be empty.")
        return search_term.strip()

class ResumeRequest(BaseModel):
    thread_id: str
    plan: Dict[str, Any]

def _dump_object(obj: Any) -> Any:
    if obj is None: return None
    if hasattr(obj, "model_dump"): return obj.model_dump()
    if hasattr(obj, "dict"): return obj.dict()
    if isinstance(obj, dict): return obj
    return str(obj)

@app.get("/")
def read_root():
    return {"status": "online", "message": "API running with HITL Support."}

async def run_and_yield_events(input_state: Any, config: dict, thread_id: str):
    try:
        async for chunk in research_graph.astream(input_state, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                
                # SAFELY unwrap tuple if LangGraph mangles the state
                if isinstance(node_state, tuple):
                    node_state = node_state[0] if len(node_state) > 0 else {}
                
                if not isinstance(node_state, dict):
                    continue

                event_payload = {
                    "thread_id": thread_id,
                    "node": node_name,
                    "status": node_state.get("status", "processing"),
                    "plan": _dump_object(node_state.get("plan")),
                    "sources": [_dump_object(s) for s in node_state.get("sources", [])],
                    "draft": _dump_object(node_state.get("draft")),
                    "fact_checks": [_dump_object(fc) for fc in node_state.get("fact_checks", [])],
                }
                yield f"data: {json.dumps(event_payload)}\n\n"
    except Exception as e:
        error_payload = {"thread_id": thread_id, "node": "error", "message": str(e)}
        yield f"data: {json.dumps(error_payload)}\n\n"

@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    try:
        topic = request.get_search_topic()
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    thread_id = request.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic": topic,
        "query": topic,
        "auto_approve": request.auto_approve,
        "plan": None,
        "sources": [],
        "draft": None,
        "fact_checks": [],
        "retry_count": 0,
        "status": "started",
    }
    
    return StreamingResponse(run_and_yield_events(initial_state, config, thread_id), media_type="text/event-stream")

@app.post("/research/resume")
async def resume_research(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}

    # FIX: Remove as_node="planner". 
    # By omitting it, LangGraph seamlessly pushes the updated plan into the pending 'human_review' node!
    await research_graph.aupdate_state(config, {"plan": request.plan})

    return StreamingResponse(run_and_yield_events(None, config, request.thread_id), media_type="text/event-stream")