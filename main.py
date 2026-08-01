import json
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import workflow, get_research_graph

# Global reference for initialized graph
research_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_graph
    # Connect asynchronously to SQLite database on startup
    async with AsyncSqliteSaver.from_conn_string("research_memory.db") as checkpointer:
        research_graph = get_research_graph(checkpointer)
        print("--- [DATABASE] AsyncSqliteSaver Checkpointer active ---")
        yield

app = FastAPI(
    title="Free AI Research Agent API",
    description="Backend API powering the LangGraph multi-agent research workflow with persistent thread memory.",
    version="3.0.0",
    lifespan=lifespan,
)

# Enable CORS so Streamlit or external dashboards can communicate seamlessly
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

    def get_search_topic(self) -> str:
        search_term = self.topic or self.query
        if not search_term or not search_term.strip():
            raise ValueError("Research topic/query cannot be empty.")
        return search_term.strip()


class ResearchResponse(BaseModel):
    thread_id: str
    topic: str
    plan: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    draft: Optional[Dict[str, Any]] = None
    fact_checks: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "completed"


def _dump_object(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return str(obj)


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "AI Research Agent API is running with Async SQLite persistence.",
    }


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
        "plan": None,
        "sources": [],
        "draft": None,
        "fact_checks": [],
        "retry_count": 0,
        "status": "started",
    }

    async def event_generator():
        try:
            async for chunk in research_graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in chunk.items():
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

    return StreamingResponse(event_generator(), media_type="text/event-stream")