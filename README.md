# 🤖 Autonomous Multi-Agent AI Research Workspace

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An agentic, multi-node research platform powered by **LangGraph**, **FastAPI**, and **Streamlit**. The system dynamically formulates research plans, executes deep web retrieval, synthesizes structured technical reports with real-time token streaming, and verifies claims using a automated fact-checking guardrail.

---

## 💡 Key Architectural Features

- **⚡ Real-Time SSE Token Streaming**: Low-latency Server-Sent Events (SSE) stream generation tokens dynamically from local/cloud LLM endpoints to the Streamlit UI.
- **✋ Human-in-the-Loop (HITL) Orchestration**: State graph persistence allows human approval and editing of AI research plans mid-flight before web queries are executed.
- **🔄 Fault-Tolerant State Machine**: Built on LangGraph state charts with SQLite checkpointers, allowing full session restoration, history tracking, and conditional retry loops.
- **🛡️ Fact-Verification Guardrail**: Automated evaluation node checks synthesized claims against retrieved web snippets and triggers dynamic synthesis retries upon failure.
- **📥 Deep-Dive Export Suite**: Generates formatted, production-ready research reports in standard `.md` (Markdown) and `.docx` (Microsoft Word) formats.
- **💬 Interactive Report RAG**: Memory-backed chat interface allowing users to query gathered source contexts for post-synthesis exploration.

---

## 🏗️ System Architecture

```mermaid
graph TD
    Start([User Query]) --> Planner[🧠 Planner Node]
    Planner --> AutoApprove{Auto Approve?}
    
    AutoApprove -- No --> HumanReview[✋ Human-in-the-Loop Review]
    HumanReview -- Plan Approved --> Retriever[🌐 Web Retriever Node]
    AutoApprove -- Yes --> Retriever
    
    Retriever --> Synthesizer[✍️ Synthesizer Node]
    Synthesizer --> FactChecker[✅ Fact Checker Guardrail]
    
    FactChecker -- Verification Failed & Retries < 1 --> Synthesizer
    FactChecker -- Verified / Max Retries --> Finished([📝 Final Report & Export])

    🛠️ Tech Stack
    Category,Technology,Purpose
Agent Framework,LangGraph / LangChain,"State Graph orchestration, structured outputs, HITL checkpoints"
Backend API,FastAPI / Uvicorn,Asynchronous REST backend with Server-Sent Events (SSE)
Frontend UI,Streamlit,"Interactive UI, session persistence, document rendering"
LLM Engine,LM Studio / OpenAI-Compatible,Local or cloud LLM inference (qwen2.5-7b-instruct)
Document Processing,python-docx,Programmatic Microsoft Word compilation
State Persistence,AsyncSqliteSaver,Asynchronous session checkpointer for graph resume

🚀 Quickstart Guide
Prerequisites
Python 3.10+

LM Studio running locally (or any OpenAI-compatible API key) with qwen2.5-7b-instruct or similar.

1. Repository Setup
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

2. Environment Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Launch Services
Terminal 1 — Backend (FastAPI):
uvicorn main:app --reload --port 8000
Terminal 2 — Frontend (Streamlit):
streamlit run app.py
Open your browser to http://localhost:8501 to access the workspace.
📋 API Reference
Endpoint,Method,Description
/research/stream,POST,Initializes agent execution and streams node events/tokens via SSE
/research/resume,POST,Resumes paused graph execution after Human-in-the-Loop plan approval
/research/chat,POST,Streams contextual Q&A answers against stored session research