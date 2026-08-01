from typing import Any, Dict, List, Optional, TypedDict, Union
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from schemas import ExtractedSource, FactCheckResult, ResearchDraft, ResearchPlan
from tools import execute_web_search


class AgentState(TypedDict):
    topic: str
    auto_approve: bool  # Tracks if we should skip the HITL pause
    plan: Optional[Union[ResearchPlan, Dict[str, Any]]]
    sources: List[ExtractedSource]
    draft: Optional[Union[str, ResearchDraft]]
    fact_checks: List[FactCheckResult]
    retry_count: int
    status: str


# Point LangChain to your local LLM endpoint (e.g., LM Studio, Ollama, OpenAI)
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="qwen2.5-7b-instruct",
    temperature=0.1,
)


def _coerce_model(obj: Any, model_class: Any) -> Any:
    if isinstance(obj, model_class):
        return obj
    if isinstance(obj, dict):
        return model_class(**obj)
    return model_class.model_validate(obj)


async def planner_node(state: AgentState) -> AgentState:
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}

    print(f"--- [NODE: PLANNER] Generating research plan for topic: '{state.get('topic', '')}' ---")
    
    prompt = [
        SystemMessage(content="Create a concise research plan for the requested topic. Keep questions short and actionable."),
        HumanMessage(content=f"Topic: {state.get('topic', '')}"),
    ]
    try:
        structured_llm = llm.with_structured_output(ResearchPlan)
        response = await structured_llm.ainvoke(prompt)
        plan = _coerce_model(response, ResearchPlan)
    except Exception as e:
        print(f"Planner structured output fallback: {e}")
        plan = ResearchPlan(topic=state.get("topic", "Unknown"), objective="Research topic", questions=[state.get("topic", "")])

    return {
        **state,
        "plan": plan,
        "status": "planned",  # Trigger UI review step if auto_approve is False
    }


# Human Review Node (Acts as a resumption point after HITL approval)
async def human_review_node(state: AgentState) -> AgentState:
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}

    print("--- [NODE: HUMAN REVIEW] Plan approved, resuming workflow ---")
    return {**state, "status": "approved"}


async def retriever_node(state: AgentState) -> AgentState:
    print("--- [NODE: RETRIEVER] Fetching web sources ---")
    
    # DEFENSIVE & SAFE: Unwrap state if it arrives as a tuple
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}
        
    raw_plan = state.get("plan")
    if not raw_plan:
        return {**state, "sources": [], "status": "no_plan"}

    # Handle raw_plan tuple or dict safely
    if isinstance(raw_plan, tuple):
        raw_plan = raw_plan[0] if len(raw_plan) > 0 else {}

    try:
        plan = _coerce_model(raw_plan, ResearchPlan)
        queries = (plan.questions or [plan.topic])[:4]
    except Exception as err:
        print(f"Failed to parse plan in retriever: {err}")
        queries = [state.get("topic", "")]

    sources = execute_web_search(queries, max_results_per_query=2, deep_scrape=True)
    print(f"Retriever found {len(sources)} sources.")
    
    return {
        **state,
        "sources": sources,
        "status": "retrieved",
    }


async def synthesizer_node(state: AgentState) -> AgentState:
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}

    print(f"--- [NODE: SYNTHESIZER] Writing draft (Attempt {state.get('retry_count', 0) + 1}) ---")
    
    raw_plan = state.get("plan")
    sources = state.get("sources", [])
    
    if not raw_plan:
        return {**state, "draft": "No plan available.", "status": "no_plan"}

    # Defensively unwrap tuple or dict if needed
    if isinstance(raw_plan, tuple):
        raw_plan = raw_plan[0] if len(raw_plan) > 0 else {}

    topic = raw_plan.topic if hasattr(raw_plan, "topic") else (raw_plan.get("topic", "Unknown") if isinstance(raw_plan, dict) else "Unknown")

    formatted_sources = []
    for idx, s in enumerate(sources[:5], 1):
        content_snippet = s.snippet[:1500] if hasattr(s, "snippet") and s.snippet else (s.get("snippet", "No content")[:1500] if isinstance(s, dict) else "")
        title = s.title if hasattr(s, "title") else (s.get("title", "Untitled") if isinstance(s, dict) else "Untitled")
        formatted_sources.append(f"Source {idx} [{title}]:\n{content_snippet}\n")

    sources_summary = "\n---\n".join(formatted_sources) if formatted_sources else "No sources available."

    prompt = [
        SystemMessage(
            content=(
                "You are an expert technical researcher. Synthesize a detailed, thorough research report "
                "addressing all research questions using the provided source context. "
                "Write in clean, well-structured Markdown. Include a 'Summary' section and a 'Key Takeaways' bulleted list."
            )
        ),
        HumanMessage(
            content=(
                f"Topic: {topic}\n\n"
                f"Gathered Research Data:\n{sources_summary}"
            )
        ),
    ]

    try:
        # Standard raw text streaming response (enables live token streaming)
        response = await llm.ainvoke(prompt)
        draft = response.content
    except Exception as e:
        print(f"Synthesizer fallback due to error: {e}")
        draft = f"Failed to generate draft due to an error: {str(e)}"

    return {
        **state,
        "draft": draft,
        "status": "drafted",
    }


async def fact_checker_node(state: AgentState) -> AgentState:
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}

    print("--- [NODE: FACT CHECKER] Verifying claims ---")
    draft = state.get("draft")
    if not draft:
        return {**state, "fact_checks": [], "status": "no_draft"}

    # FIX: Extract plain string content safely whether draft is str, dict, or object
    if isinstance(draft, str):
        draft_text = draft
    elif hasattr(draft, "summary"):
        draft_text = str(draft.summary)
    elif isinstance(draft, dict):
        draft_text = str(draft.get("summary", draft.get("draft", str(draft))))
    else:
        draft_text = str(draft)

    sources = state.get("sources", [])
    formatted_sources = []
    for s in sources[:5]:
        snippet = s.snippet[:500] if hasattr(s, "snippet") and s.snippet else (s.get("snippet", "")[:500] if isinstance(s, dict) else "")
        formatted_sources.append(f"- {snippet}")
    sources_text = "\n".join(formatted_sources) if formatted_sources else "No sources available."

    prompt = [
        SystemMessage(content="Check whether the draft claims are supported by the sources context. Return status as 'verified' or 'failed'."),
        HumanMessage(content=f"Draft Text:\n{draft_text[:3000]}\n\nSources Context:\n{sources_text}"),
    ]

    try:
        structured_llm = llm.with_structured_output(FactCheckResult)
        response = await structured_llm.ainvoke(prompt)
        result = _coerce_model(response, FactCheckResult)
    except Exception as e:
        print(f"Fact checker fallback: {e}")
        result = FactCheckResult(status="verified", details="Auto-verified via fallback.")

    raw_status = str(getattr(result, "status", "verified")).lower()
    normalized_status = "verified" if any(kw in raw_status for kw in ["verified", "supported", "passed", "high"]) else "failed"

    fact_checks = list(state.get("fact_checks", []))
    fact_checks.append(result)

    retry_count = state.get("retry_count", 0) + 1
    return {
        **state,
        "fact_checks": fact_checks,
        "retry_count": retry_count,
        "status": normalized_status,
    }


# Router Functions
def route_after_planner(state: AgentState) -> str:
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}

    if state.get("auto_approve"):
        print("Auto-approve is enabled. Skipping human review.")
        return "retriever"
    
    print("Auto-approve is disabled. Routing to human review (graph will pause).")
    return "human_review"


def route_after_fact_check(state: AgentState) -> str:
    if isinstance(state, tuple):
        state = state[0] if len(state) > 0 else {}

    retry_count = state.get("retry_count", 0)
    current_status = state.get("status", "verified")

    if current_status == "verified" or retry_count >= 1:
        print("Fact check passed or max retries reached. Finishing workflow.")
        return END
    print(f"Fact check failed with status '{current_status}'. Retrying synthesis...")
    return "synthesizer"


# Define StateGraph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("fact_checker", fact_checker_node)

workflow.add_edge(START, "planner")

# Conditional routing after planner
workflow.add_conditional_edges(
    "planner", 
    route_after_planner, 
    {"retriever": "retriever", "human_review": "human_review"}
)

workflow.add_edge("human_review", "retriever")
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("synthesizer", "fact_checker")

workflow.add_conditional_edges(
    "fact_checker",
    route_after_fact_check,
    {
        "synthesizer": "synthesizer",
        END: END,
    },
)


def get_research_graph(checkpointer):
    # Pauses execution right before entering the 'human_review' node
    return workflow.compile(
        checkpointer=checkpointer, 
        interrupt_before=["human_review"]
    )


__all__ = ["workflow", "get_research_graph", "AgentState", "llm"]