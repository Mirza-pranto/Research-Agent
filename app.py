import requests
import streamlit as st


st.set_page_config(page_title="AI Research Dashboard", layout="wide")

st.title("Free AI Research Agent")
st.write("Submit a research prompt to run the local FastAPI workflow.")

prompt = st.text_area("Research prompt", placeholder="Enter a topic or question...")

if st.button("Run research") and prompt.strip():
    with st.spinner("Running research workflow..."):
        response = requests.post(
            "http://localhost:8000/research",
            json={"topic": prompt.strip()},
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()

    st.subheader("Summary")
    draft = data.get("draft") or {}
    st.write(draft.get("summary") or "No summary available yet.")

    st.subheader("Key takeaways")
    key_points = draft.get("key_points") or []
    if key_points:
        for point in key_points:
            st.markdown(f"- {point}")
    else:
        st.info("No key takeaways returned.")

    st.subheader("Fact check status")
    fact_checks = data.get("fact_checks") or []
    if fact_checks:
        for check in fact_checks:
            st.write(f"- {check.get('claim', 'N/A')}: {check.get('status', 'unknown')}")
    else:
        st.info("No fact-check results available.")

    st.subheader("Revision count")
    st.write(data.get("status") or "unknown")

    st.subheader("Sources")
    sources = data.get("sources") or []
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
                st.write(snippet)
    else:
        st.info("No sources returned.")
