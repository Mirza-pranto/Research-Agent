import json
import httpx
import streamlit as st

st.set_page_config(page_title="AI Research Dashboard", layout="wide")

st.title("Free AI Research Agent")
st.write("Submit a research prompt to run the local FastAPI workflow with real-time step streaming.")

prompt = st.text_area("Research prompt", placeholder="Enter a topic or question...")

if st.button("Run research") and prompt.strip():
    # Progress status container for streaming steps
    status_box = st.status("🚀 Initializing research workflow...", expanded=True)

    # Dictionary to hold the aggregated state from SSE events
    final_data = {
        "plan": None,
        "sources": [],
        "draft": None,
        "fact_checks": [],
        "status": "started",
    }

    try:
        url = "http://localhost:8000/research/stream"
        
        # Use httpx to consume Server-Sent Events (SSE)
        with httpx.stream("POST", url, json={"topic": prompt.strip()}, timeout=600.0) as response:
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

    # Render Final Results Section
    st.divider()

    st.subheader("Summary")
    draft = final_data.get("draft") or {}
    st.write(draft.get("summary") or "No summary available yet.")

    st.subheader("Key takeaways")
    key_points = draft.get("key_points") or []
    if key_points:
        for point in key_points:
            st.markdown(f"- {point}")
    else:
        st.info("No key takeaways returned.")

    st.subheader("Fact check status")
    fact_checks = final_data.get("fact_checks") or []
    if fact_checks:
        for check in fact_checks:
            claim = check.get("claim") or check.get("details") or "N/A"
            status_val = check.get("status") or "unknown"
            st.write(f"- {claim}: **{status_val}**")
    else:
        st.info("No fact-check results available.")

    st.subheader("Revision count / Status")
    st.write(final_data.get("status") or "unknown")

    st.subheader("Sources")
    sources = final_data.get("sources") or []
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
                st.caption(snippet[:250] + "..." if len(snippet) > 250 else snippet)
    else:
        st.info("No sources returned.")