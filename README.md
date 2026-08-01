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