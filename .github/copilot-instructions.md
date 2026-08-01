# GitHub Copilot Instructions - Free AI Research Agent

## Project Stack
- Python 3.10+
- Orchestration: LangGraph & LangChain
- Local LLM Engine: ChatOllama (model="qwen2.5:7b")
- Web Search Tool: DuckDuckGoSearchRun (Free, no API keys)
- API: FastAPI & Uvicorn
- UI: Streamlit

## Architecture Rules
1. State Management: Use `TypedDict` for `AgentState` in `graph.py`.
2. Output Validation: All LLM nodes producing structured output must use Pydantic models.
3. Cost Constraint: Do NOT use paid APIs like OpenAI or Tavily. Always use `ChatOllama` and `DuckDuckGoSearchRun`.
4. File Separation:
   - `schemas.py`: Pydantic models only.
   - `tools.py`: Search functions only.
   - `graph.py`: LangGraph nodes, edges, and graph compilation.
   - `main.py`: FastAPI routes.
   - `app.py`: Streamlit frontend.