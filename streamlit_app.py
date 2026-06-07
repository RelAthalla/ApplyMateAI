"""Simple Streamlit frontend for ApplyMate AI."""

from __future__ import annotations

import streamlit as st

from app.llm import MissingAPIKeyError
from app.service import ApplyMateService


st.set_page_config(page_title="ApplyMate AI", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
      .block-container {max-width: 1000px; padding-top: 2rem; padding-bottom: 3rem;}
      .app-card {padding: 1rem 1.2rem; border: 1px solid #e5e7eb; border-radius: 14px; background: #ffffff;}
      .muted {color: #475569;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_streamlit_service() -> ApplyMateService:
    return ApplyMateService()


st.title("ApplyMate AI")
st.caption("Upload a CV, paste a job description, and generate a grounded application report.")

st.markdown('<div class="app-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload CV file", type=["pdf", "txt"])
job_description = st.text_area("Job description", height=240, placeholder="Paste the target role description here...")
analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


if analyze_clicked:
    if uploaded_file is None:
        st.error("Please upload a CV file before analyzing.")
    elif not job_description.strip():
        st.error("Please paste a job description before analyzing.")
    else:
        try:
            service = get_streamlit_service()
            with st.spinner("Analyzing your CV against the job description..."):
                report = service.analyze_upload(
                    filename=uploaded_file.name,
                    file_bytes=uploaded_file.getvalue(),
                    job_description=job_description,
                )
        except MissingAPIKeyError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader("Results")
            score_col, title_col = st.columns([1, 3])
            score_col.metric("Match Score", f"{report.match_score}/100")
            title_col.markdown(f"**Target Role:** {report.role_title}")
            st.write(report.summary)

            left_col, right_col = st.columns(2)
            with left_col:
                st.markdown("### Strengths")
                for item in report.strengths:
                    st.markdown(f"- {item}")

                st.markdown("### Missing Skills")
                for item in report.missing_skills:
                    st.markdown(f"- {item}")

            with right_col:
                st.markdown("### Relevant CV Evidence")
                for item in report.relevant_cv_evidence:
                    st.markdown(f"- {item}")

            st.markdown("### Retrieved CV Snippets")
            for item in report.relevant_experience:
                with st.expander(f"{item.chunk_id} | {item.relevance_reason}"):
                    st.write(item.snippet)

            st.markdown("### Recommended Projects")
            for project in report.recommended_projects:
                with st.container(border=True):
                    st.markdown(f"**{project.project_title}**")
                    st.write(project.short_description)
                    st.markdown(f"**Tech stack:** {', '.join(project.tech_stack)}")
                    st.markdown(f"**Why it helps:** {project.why_it_helps}")
                    st.markdown("**MVP features**")
                    for feature in project.mvp_features:
                        st.markdown(f"- {feature}")

            st.markdown("### Cover Letter")
            st.markdown(f"**{report.cover_letter.headline}**")
            st.write(report.cover_letter.body)

            st.markdown("### Interview Questions")
            for index, question in enumerate(report.interview_questions, start=1):
                with st.expander(f"{index}. {question.question}"):
                    st.write(question.suggested_answer)

            st.download_button(
                label="Download Markdown Report",
                data=report.markdown_report,
                file_name="applymate_report.md",
                mime="text/markdown",
            )
