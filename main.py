import json
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from graph import workflow, get_research_graph, llm # Make sure to import llm from graph
from langchain_core.messages import SystemMessage, HumanMessage

research_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global research_graph
    async with AsyncSqliteSaver.from_conn_string("research_memory.db") as checkpointer:
        research_graph = get_research_graph(checkpointer)
        print("--- [DATABASE] AsyncSqliteSaver Checkpointer active ---")
        yield

app = FastAPI(title="AI Research Agent API", version="5.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class ResearchRequest(BaseModel):
    topic: Optional[str] = None
    query: Optional[str] = None
    thread_id: Optional[str] = None
    auto_approve: bool = True

class ResumeRequest(BaseModel):
    thread_id: str
    plan: Dict[str, Any]

class ChatRequest(BaseModel):
    thread_id: str
    message: str

def _dump_object(obj: Any) -> Any:
    if obj is None: return None
    if hasattr(obj, "model_dump"): return obj.model_dump()
    if hasattr(obj, "dict"): return obj.dict()
    if isinstance(obj, dict): return obj
    return str(obj)

async def run_and_yield_events(input_state: Any, config: dict, thread_id: str):
    try:
        # UPGRADE: stream_mode=["messages", "updates"] captures live LLM tokens AND node state updates
        async for event_type, chunk in research_graph.astream(input_state, config=config, stream_mode=["messages", "updates"]):
            
            # 1. Handle live tokens from the synthesizer
            if event_type == "messages":
                chunk_msg, metadata = chunk
                if metadata.get("langgraph_node") == "synthesizer" and chunk_msg.content:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk_msg.content})}\n\n"
            
            # 2. Handle state updates
            elif event_type == "updates":
                for node_name, node_state in chunk.items():
                    if isinstance(node_state, tuple):
                        node_state = node_state[0] if len(node_state) > 0 else {}
                    if not isinstance(node_state, dict):
                        continue

                    event_payload = {
                        "type": "update",
                        "thread_id": thread_id,
                        "node": node_name,
                        "status": node_state.get("status", "processing"),
                        "plan": _dump_object(node_state.get("plan")),
                        "sources": [_dump_object(s) for s in node_state.get("sources", [])],
                        "draft": _dump_object(node_state.get("draft")),
                    }
                    yield f"data: {json.dumps(event_payload)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'update', 'node': 'error', 'message': str(e)})}\n\n"

@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    thread_id = request.thread_id or str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "topic": request.topic,
        "query": request.topic,
        "auto_approve": request.auto_approve,
        "plan": None, "sources": [], "draft": None, "fact_checks": [], "retry_count": 0, "status": "started",
    }
    return StreamingResponse(run_and_yield_events(initial_state, config, thread_id), media_type="text/event-stream")

@app.post("/research/resume")
async def resume_research(request: ResumeRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    await research_graph.aupdate_state(config, {"plan": request.plan})
    return StreamingResponse(run_and_yield_events(None, config, request.thread_id), media_type="text/event-stream")

# ==========================================
# UPGRADE: Chat with your Research (RAG)
# ==========================================
@app.post("/research/chat")
async def chat_research(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # 1. Load the exact graph state for this thread directly from the database
    state = await research_graph.aget_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session not found.")
    
    sources = state.values.get("sources", [])
    
    # 2. Format the retrieved sources as context
    formatted_sources = []
    for s in sources[:5]:
        title = s.get("title") if isinstance(s, dict) else getattr(s, "title", "Unknown")
        snippet = s.get("snippet") if isinstance(s, dict) else getattr(s, "snippet", "")
        formatted_sources.append(f"Source [{title}]: {snippet}")
    context = "\n".join(formatted_sources) if formatted_sources else "No web sources available."

    # 3. Stream the LLM response
    async def generate_chat():
        prompt = [
            SystemMessage(content=f"You are a helpful AI assistant. Answer the user's question using ONLY the following research context. If the answer is not in the context, say so.\n\nCONTEXT:\n{context}"),
            HumanMessage(content=request.message)
        ]
        async for chunk in llm.astream(prompt):
            if chunk.content:
                yield chunk.content
                
    return StreamingResponse(generate_chat(), media_type="text/plain")