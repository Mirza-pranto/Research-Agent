import json
import uuid
import httpx
import streamlit as st

st.set_page_config(page_title="AI Research Workspace", layout="wide")

# Initialize session state for multi-thread history
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # {thread_id: {"topic": ..., "data": ...}}

if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = str(uuid.uuid4())

active_thread = st.session_state.active_thread_id
current_session_data = st.session_state.sessions.get(active_thread, {})

# Sidebar Navigation for Research Sessions
with st.sidebar:
    st.title("📚 Research History")

    if st.button("➕ New Research Session", use_container_width=True):
        st.session_state.active_thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    if st.session_state.sessions:
        st.write("**Saved Sessions:**")
        for tid, sess in st.session_state.sessions.items():
            topic_label = sess.get("topic", "Untitled Session")
            truncated_label = topic_label[:26] + ("..." if len(topic_label) > 26 else "")

            # Highlight currently active session
            is_active = (tid == st.session_state.active_thread_id)
            btn_prefix = "🟢 " if is_active else "📄 "

            if st.button(f"{btn_prefix}{truncated_label}", key=f"btn_{tid}", use_container_width=True):
                st.session_state.active_thread_id = tid
                st.rerun()
    else:
        st.caption("No past sessions yet. Submit a research topic to start!")

st.title("Free AI Research Agent Workspace 3.0")
st.write("Submit a research prompt to run the local FastAPI workflow with real-time streaming and thread memory.")

# Pre-fill input box if active session already has a topic
default_prompt = current_session_data.get("topic", "")
prompt = st.text_area("Research prompt", value=default_prompt, placeholder="Enter a topic or question...", key=f"prompt_{active_thread}")

if st.button("Run research", key=f"run_{active_thread}") and prompt.strip():
    status_box = st.status("🚀 Initializing research workflow...", expanded=True)

    final_data = {
        "topic": prompt.strip(),
        "plan": None,
        "sources": [],
        "draft": None,
        "fact_checks": [],
        "status": "started",
    }

    try:
        url = "http://localhost:8000/research/stream"
        payload = {
            "topic": prompt.strip(),
            "thread_id": active_thread,
        }

        with httpx.stream("POST", url, json=payload, timeout=600.0) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                raw_json = line[6:].strip()
                if not raw_json:
                    continue

                event = json.loads(raw_json)
                node = event.get("node")

                if node == "planner":
                    status_box.update(label="🧠 **Planner:** Research plan generated!", state="running")
                    final_data["plan"] = event.get("plan")
                    if final_data["plan"]:
                        with status_box:
                            st.write("**Target Questions:**")
                            for q in final_data["plan"].get("questions", []):
                                st.write(f"- {q}")

                elif node == "retriever":
                    status_box.update(label="🌐 **Retriever:** Web sources collected!", state="running")
                    final_data["sources"] = event.get("sources", [])

                elif node == "synthesizer":
                    status_box.update(label="✍️ **Synthesizer:** Research draft written!", state="running")
                    final_data["draft"] = event.get("draft")

                elif node == "fact_checker":
                    status_box.update(label="✅ **Fact Checker:** Claim verification complete!", state="complete")
                    final_data["fact_checks"] = event.get("fact_checks", [])
                    final_data["status"] = event.get("status", "completed")

                elif node == "error":
                    status_box.update(label="❌ Error occurred in workflow", state="error")
                    st.error(f"Agent Error: {event.get('message')}")

    except Exception as err:
        status_box.update(label="❌ Connection failed", state="error")
        st.error(f"Failed to connect to streaming backend: {err}")

    # Store full research session in Streamlit state
    st.session_state.sessions[active_thread] = final_data
    current_session_data = final_data

# Render Active Session Results if present
if current_session_data and current_session_data.get("status") in ["completed", "verified", "drafted"]:
    st.divider()

    st.subheader("Summary")
    draft = current_session_data.get("draft") or {}
    st.write(draft.get("summary") or "No summary available yet.")

    st.subheader("Key takeaways")
    key_points = draft.get("key_points") or []
    if key_points:
        for point in key_points:
            st.markdown(f"- {point}")
    else:
        st.info("No key takeaways returned.")

    st.subheader("Fact check status")
    fact_checks = current_session_data.get("fact_checks") or []
    if fact_checks:
        for check in fact_checks:
            claim = check.get("claim") or check.get("details") or "N/A"
            status_val = check.get("status") or "unknown"
            st.write(f"- {claim}: **{status_val}**")
    else:
        st.info("No fact-check results available.")

    st.subheader("Revision count / Status")
    st.write(current_session_data.get("status") or "unknown")

    st.subheader("Sources")
    sources = current_session_data.get("sources") or []
    if sources:
        for source in sources:
            title = source.get("title") or "Untitled"
            url = source.get("url")
            snippet = source.get("snippet") or ""
            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.markdown(f"- {title}")
            if snippet:
                st.caption(snippet[:300] + "..." if len(snippet) > 300 else snippet)
    else:
        st.info("No sources returned.")