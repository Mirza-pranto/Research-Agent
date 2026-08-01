import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from graph import research_graph

app = FastAPI(
    title="Free AI Research Agent API",
    description="Backend API powering the LangGraph multi-agent research workflow.",
    version="2.0.0",
)

# Enable CORS so Streamlit or external dashboards can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schema (Supports both 'topic' and 'query' to avoid payload mismatches)
class ResearchRequest(BaseModel):
    topic: Optional[str] = None
    query: Optional[str] = None

    def get_search_topic(self) -> str:
        """Utility to pull whichever field Streamlit sent."""
        search_term = self.topic or self.query
        if not search_term or not search_term.strip():
            raise ValueError("Research topic/query cannot be empty.")
        return search_term.strip()


# Response schema
class ResearchResponse(BaseModel):
    topic: str
    plan: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    draft: Optional[Dict[str, Any]] = None
    fact_checks: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "completed"


def _dump_object(obj: Any) -> Any:
    """Helper function to convert Pydantic objects or dicts into standard dicts."""
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
    """Health check endpoint."""
    return {
        "status": "online",
        "message": "AI Research Agent API is running. POST to /research or /research/stream to start a job.",
    }


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest) -> ResearchResponse:
    """Synchronous research endpoint (Wait for full completion)."""
    try:
        topic = request.get_search_topic()
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    # Construct initial state payload for LangGraph
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

    try:
        # Run state graph
        result = research_graph.invoke(initial_state)

        # Convert state items safely to dicts
        plan_data = _dump_object(result.get("plan"))
        draft_data = _dump_object(result.get("draft"))

        raw_sources = result.get("sources", [])
        sources_data = [_dump_object(src) for src in raw_sources]

        raw_fact_checks = result.get("fact_checks", [])
        fact_checks_data = [_dump_object(fc) for fc in raw_fact_checks]

        return ResearchResponse(
            topic=topic,
            plan=plan_data,
            sources=sources_data,
            draft=draft_data,
            fact_checks=fact_checks_data,
            status=result.get("status", "completed"),
        )

    except Exception as e:
        print(f"Error executing research graph: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal Agent Graph Error: {str(e)}"
        )


@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    """Real-time SSE endpoint streaming node updates as they complete."""
    try:
        topic = request.get_search_topic()
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

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
            # Stream graph updates node-by-node using LangGraph's .astream()
            async for chunk in research_graph.astream(initial_state, stream_mode="updates"):
                for node_name, node_state in chunk.items():
                    event_payload = {
                        "node": node_name,
                        "status": node_state.get("status", "processing"),
                        "plan": _dump_object(node_state.get("plan")),
                        "sources": [_dump_object(s) for s in node_state.get("sources", [])],
                        "draft": _dump_object(node_state.get("draft")),
                        "fact_checks": [_dump_object(fc) for fc in node_state.get("fact_checks", [])],
                    }
                    # Send payload formatted as Server-Sent Event (SSE)
                    yield f"data: {json.dumps(event_payload)}\n\n"
        except Exception as e:
            error_payload = {"node": "error", "message": str(e)}
            yield f"data: {json.dumps(error_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")