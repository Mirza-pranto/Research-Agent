import json
import uuid
import httpx
import streamlit as st

st.set_page_config(page_title="AI Research Workspace", layout="wide")

if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = str(uuid.uuid4())

active_thread = st.session_state.active_thread_id
current_session_data = st.session_state.sessions.get(active_thread, {})

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
            
            is_active = (tid == st.session_state.active_thread_id)
            btn_prefix = "🟢 " if is_active else "📄 "

            if st.button(f"{btn_prefix}{truncated_label}", key=f"btn_{tid}", use_container_width=True):
                st.session_state.active_thread_id = tid
                st.rerun()


st.title("Free AI Research Agent Workspace 4.0")
st.write("Now with **Human-in-the-Loop (HITL)** Plan Approval!")

default_prompt = current_session_data.get("topic", "")
prompt = st.text_area("Research prompt", value=default_prompt, placeholder="Enter a topic...", key=f"prompt_{active_thread}")

# NEW: Toggle for auto-approving the plan
auto_approve = st.toggle("Auto-approve research plan (Skip manual review)", value=True)

# Helper function to consume the SSE stream and update UI
def consume_stream(url: str, payload: dict, final_data: dict, status_box):
    has_error = False  # Track errors to prevent blind reruns
    
    try:
        with httpx.stream("POST", url, json=payload, timeout=600.0) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line or not line.startswith("data: "): continue
                raw_json = line[6:].strip()
                if not raw_json: continue

                event = json.loads(raw_json)
                node = event.get("node")
                
                # FIX: Universally update the status on every successful event
                new_status = event.get("status")
                if new_status and new_status != "processing":
                    final_data["status"] = new_status

                if node == "planner":
                    status_box.update(label="🧠 **Planner:** Research plan generated!", state="running")
                    final_data["plan"] = event.get("plan")
                    
                elif node == "human_review":
                    status_box.update(label="👍 **Human Review:** Plan approved, resuming...", state="running")

                elif node == "retriever":
                    status_box.update(label="🌐 **Retriever:** Web sources collected!", state="running")
                    final_data["sources"] = event.get("sources", [])

                elif node == "synthesizer":
                    status_box.update(label="✍️ **Synthesizer:** Research draft written!", state="running")
                    final_data["draft"] = event.get("draft")

                elif node == "fact_checker":
                    status_box.update(label="✅ **Fact Checker:** Claim verification complete!", state="complete")
                    final_data["fact_checks"] = event.get("fact_checks", [])

                elif node == "error":
                    status_box.update(label="❌ Error occurred in workflow", state="error")
                    st.error(f"Agent Error: {event.get('message')}")
                    has_error = True
                    
    except Exception as err:
        status_box.update(label="❌ Connection failed", state="error")
        st.error(f"Failed to connect to streaming backend: {err}")
        has_error = True
    
    return final_data, has_error


# PRIMARY RUN BUTTON
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
    
    payload = {
        "topic": prompt.strip(),
        "thread_id": active_thread,
        "auto_approve": auto_approve
    }

    # Only stream the primary run here!
    final_data, has_error = consume_stream("http://localhost:8000/research/stream", payload, final_data, status_box)
    st.session_state.sessions[active_thread] = final_data
    
    if not has_error:
        st.rerun()


# ==========================================
# INTERRUPT STATE UI: Plan Editing Form
# ==========================================
if current_session_data.get("status") == "planned":
    st.warning("✋ **Graph Paused:** Please review and edit the research questions before scraping begins.")
    
    plan_dict = current_session_data.get("plan", {})
    questions = plan_dict.get("questions", [])
    
    with st.form(key=f"approve_form_{active_thread}"):
        st.write("Edit, add, or delete questions below (one per line):")
        
        # Display questions as text block for easy editing
        questions_text = "\n".join(questions)
        edited_qs_text = st.text_area("Target Search Questions", value=questions_text, height=150)
        
        if st.form_submit_button("Approve & Resume Search", type="primary"):
            # Update the plan locally
            edited_questions = [q.strip() for q in edited_qs_text.split("\n") if q.strip()]
            plan_dict["questions"] = edited_questions
            current_session_data["plan"] = plan_dict
            
            status_box = st.status("▶️ Resuming workflow...", expanded=True)
            
            # Hit the Resume API endpoint
            payload = {
                "thread_id": active_thread,
                "plan": plan_dict
            }
            
            # Resume consuming the stream - Unpacking correctly applied here!
            current_session_data, has_error = consume_stream("http://localhost:8000/research/resume", payload, current_session_data, status_box)
            st.session_state.sessions[active_thread] = current_session_data
            
            if not has_error:
                st.rerun()


# ==========================================
# COMPLETED STATE UI: Final Report
# ==========================================
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