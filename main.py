from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph import research_graph

app = FastAPI(
    title="Free AI Research Agent API",
    description="Backend API powering the LangGraph multi-agent research workflow.",
    version="1.0.0",
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
        "message": "AI Research Agent API is running. POST to /research to start a job.",
    }


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest) -> ResearchResponse:
    """Main research endpoint invoked by Streamlit."""
    try:
        topic = request.get_search_topic()
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    # Construct state payload for LangGraph
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

        # Convert Pydantic state items safely to dicts
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