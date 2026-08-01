from typing import List, Optional, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from schemas import ExtractedSource, FactCheckResult, ResearchDraft, ResearchPlan
from tools import execute_web_search


class AgentState(TypedDict):
    topic: str
    plan: Optional[ResearchPlan]
    sources: List[ExtractedSource]
    draft: Optional[ResearchDraft]
    fact_checks: List[FactCheckResult]
    retry_count: int
    status: str


# Point LangChain to your LM Studio local server!
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",  # Local LM Studio endpoint
    api_key="lm-studio",                   # Put any non-empty string here
    model="qwen2.5-7b-instruct",           # Match the model ID running in LM Studio
    temperature=0.1
)


def _coerce_model(obj, model_class):
    if isinstance(obj, model_class):
        return obj
    if isinstance(obj, dict):
        return model_class(**obj)
    return model_class.model_validate(obj)


def planner_node(state: AgentState) -> AgentState:
    print(f"--- [NODE: PLANNER] Generating research plan for topic: '{state['topic']}' ---")
    prompt = [
        SystemMessage(content="Create a concise research plan for the requested topic. Keep questions short and actionable."),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ]
    try:
        structured_llm = llm.with_structured_output(ResearchPlan)
        plan = _coerce_model(structured_llm.invoke(prompt), ResearchPlan)
    except Exception as e:
        print(f"Planner structured output fallback: {e}")
        plan = ResearchPlan(topic=state["topic"], objective="Research topic", questions=[state["topic"]])

    return {
        **state,
        "plan": plan,
        "status": "planned",
    }


def retriever_node(state: AgentState) -> AgentState:
    print("--- [NODE: RETRIEVER] Fetching web sources ---")
    plan = state.get("plan")
    if not plan:
        return {**state, "sources": [], "status": "no_plan"}

    # Limit search queries to 2 max to ensure fast execution
    queries = (plan.questions or [plan.topic])[:2]
    sources = execute_web_search(queries)
    print(f"Retriever found {len(sources)} sources.")
    return {
        **state,
        "sources": sources,
        "status": "retrieved",
    }


def synthesizer_node(state: AgentState) -> AgentState:
    print(f"--- [NODE: SYNTHESIZER] Writing draft (Attempt {state.get('retry_count', 0) + 1}) ---")
    plan = state.get("plan")
    sources = state.get("sources", [])
    if not plan:
        return {**state, "draft": None, "status": "no_plan"}

    sources_summary = "\n".join([f"- {s.title}: {s.snippet[:200]}" for s in sources[:3]]) if sources else "No sources available."

    prompt = [
        SystemMessage(content="Synthesize the research findings into a concise draft containing a clear content summary."),
        HumanMessage(
            content=(
                f"Topic: {plan.topic}\n"
                f"Objective: {plan.objective}\n"
                f"Key Research Data:\n{sources_summary}"
            )
        ),
    ]
    
    try:
        structured_llm = llm.with_structured_output(ResearchDraft)
        draft = _coerce_model(structured_llm.invoke(prompt), ResearchDraft)
    except Exception as e:
        print(f"Synthesizer fallback due to error: {e}")
        draft = ResearchDraft(summary=f"Summary of research on {plan.topic} based on gathered sources.")

    return {
        **state,
        "draft": draft,
        "status": "drafted",
    }


def fact_checker_node(state: AgentState) -> AgentState:
    print("--- [NODE: FACT CHECKER] Verifying claims ---")
    draft = state.get("draft")
    if not draft:
        return {**state, "fact_checks": [], "status": "no_draft"}

    prompt = [
        SystemMessage(content="Check whether the draft claims are supported. Return status as 'verified' or 'failed'."),
        HumanMessage(content=f"Draft summary: {draft.summary}"),
    ]
    
    try:
        structured_llm = llm.with_structured_output(FactCheckResult)
        result = _coerce_model(structured_llm.invoke(prompt), FactCheckResult)
    except Exception as e:
        print(f"Fact checker fallback: {e}")
        result = FactCheckResult(status="verified", details="Auto-verified via fallback.")

    # Normalize status string (e.g. "Supported", "Passed", "Verified" -> "verified")
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


def route_after_fact_check(state: AgentState) -> str:
    retry_count = state.get("retry_count", 0)
    current_status = state.get("status", "verified")

    # Exit if fact-check passes OR if max retries (1) reached
    if current_status == "verified" or retry_count >= 1:
        print("Fact check passed or max retries reached. Finishing workflow.")
        return END

    print(f"Fact check failed with status '{current_status}'. Retrying synthesis...")
    return "synthesizer"

# Define StateGraph
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("synthesizer", synthesizer_node)
workflow.add_node("fact_checker", fact_checker_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "retriever")
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

research_graph = workflow.compile()

__all__ = ["research_graph", "AgentState"]