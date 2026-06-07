# ApplyMate AI

ApplyMate AI is a lean MVP for analyzing a candidate CV against a target job description. It combines FastAPI, LangGraph, LangChain-style retrieval components, and OpenRouter-backed LLM calls to produce a grounded match report, skill-gap analysis, project ideas, a cover letter, and interview prep.

## Features

- Upload a CV as PDF or TXT
- Paste a target job description
- Run an explicit LangGraph workflow with clear state and nodes
- Extract structured job requirements
- Retrieve relevant CV evidence before generation
- Generate a match score, strengths, weaknesses, and missing skills
- Recommend realistic portfolio projects
- Draft a tailored, evidence-grounded cover letter
- Generate interview questions and suggested answers
- Use FastAPI for the backend API
- Use Streamlit for a simple local frontend

## Tech Stack

- Python 3.11+
- FastAPI
- LangGraph
- LangChain components for chunking and retrieval plumbing
- OpenRouter API via the OpenAI-compatible Python client
- Local deterministic embeddings with optional FAISS integration when available
- Pydantic
- pypdf
- python-dotenv
- Streamlit
- pytest

## Architecture

```text
                +----------------------+
                |   Streamlit UI       |
                |  streamlit_app.py    |
                +----------+-----------+
                           |
                           | direct service call
                           v
                +----------------------+
                |   ApplyMateService   |
                |    app/service.py    |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    LangGraph Flow    |
                |     app/graph.py     |
                +----------+-----------+
                           |
         +-----------------+------------------+
         |                                    |
         v                                    v
+----------------------+          +----------------------+
|  document_loader.py  |          |     retriever.py     |
| PDF/TXT extraction   |          | chunk + retrieve CV  |
+----------------------+          +----------------------+
                           |
                           v
                +----------------------+
                |       llm.py         |
                | OpenRouter chat API  |
                +----------------------+
```

## LangGraph Workflow

The workflow is intentionally simple, explicit, and stateful:

1. `parse_cv_node`
2. `analyze_job_description_node`
3. `build_retriever_node`
4. `retrieve_relevant_experience_node`
5. `generate_match_analysis_node`
6. `generate_project_recommendations_node`
7. `generate_cover_letter_node`
8. `generate_interview_prep_node`
9. `final_report_node`

This keeps the LangGraph usage easy to follow while still demonstrating a clear multi-step orchestration pattern.

## Repository Structure

```text
applymate-ai/
|-- app/
|   |-- __init__.py
|   |-- document_loader.py
|   |-- graph.py
|   |-- llm.py
|   |-- main.py
|   |-- prompts.py
|   |-- retriever.py
|   |-- schemas.py
|   |-- service.py
|   `-- utils.py
|-- sample_data/
|   |-- sample_cv.txt
|   `-- sample_job_description.txt
|-- tests/
|   |-- test_document_loader.py
|   |-- test_graph_smoke.py
|   `-- test_schemas.py
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- run.py
`-- streamlit_app.py
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Create your environment file.

```powershell
Copy-Item .env.example .env
```

4. Add your OpenRouter credentials to `.env`.

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Running FastAPI

Start the API with:

```powershell
python run.py
```

Or directly with uvicorn:

```powershell
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Running Streamlit

The Streamlit app keeps the FastAPI backend in the project, but for simplicity it reuses the internal Python workflow directly instead of making HTTP calls.

Run it with:

```powershell
streamlit run streamlit_app.py
```

The UI includes:

- Page title: `ApplyMate AI`
- CV upload
- Job description text area
- Analyze button
- Loading spinner
- Match score
- Strengths
- Missing skills
- Relevant CV evidence
- Recommended projects
- Cover letter
- Interview questions

## API Endpoints

### `GET /`

Returns basic project information.

### `GET /health`

Returns:

```json
{"status": "ok"}
```

### `POST /analyze`

Accepts multipart form data:

- `cv_file`
- `job_description`

### `POST /analyze-text`

Accepts JSON:

```json
{
  "cv_text": "Student developer with Python and FastAPI experience.",
  "job_description": "Looking for an AI Engineer Intern with Python, FastAPI, and LangGraph."
}
```

## Example curl Request

```powershell
curl -X POST "http://127.0.0.1:8000/analyze-text" `
  -H "Content-Type: application/json" `
  -d "{\"cv_text\":\"Student developer with Python and FastAPI experience.\",\"job_description\":\"Looking for an AI Engineer Intern with Python, FastAPI, LangGraph, and RAG experience.\"}"
```

## Example Output

```json
{
  "role_title": "AI Engineer Intern",
  "summary": "Strong Python and API foundations with relevant early AI workflow experience.",
  "match_score": 81,
  "strengths": [
    "Python fundamentals",
    "FastAPI project work",
    "Hands-on prototype building"
  ],
  "missing_skills": [
    "Deeper LangGraph production usage",
    "Formal evaluation workflows"
  ],
  "relevant_cv_evidence": [
    "Built a FastAPI chatbot API for a campus FAQ assistant.",
    "Created a document search prototype with retrieval."
  ]
}
```

## Testing

Run the test suite with:

```powershell
python -m pytest
```

## Sample Data

You can test quickly with:

- `sample_data/sample_cv.txt`
- `sample_data/sample_job_description.txt`

One fast route is to copy both text files into the `/analyze-text` endpoint, or upload the sample CV in Streamlit and paste the sample job description.

## Notes on Retrieval

The retriever uses LangChain document splitting and a deterministic local embedding approach so the MVP can run without an additional embedding API. If FAISS is installed in the environment, the app will use it automatically; otherwise it falls back to a lightweight local similarity search.

## Future Improvements

- Add persistent FAISS or Chroma storage
- Add richer PDF parsing for resumes with complex layouts
- Add response caching and request tracing
- Add evaluation metrics for retrieval quality and output quality
- Add support for multiple uploaded documents such as CV plus portfolio
- Add export options for JSON, Markdown, and DOCX

## Portfolio Description

"Built an LLM-powered job application assistant using Python, FastAPI, LangGraph, LangChain, OpenRouter API, and local vector retrieval. The system analyzes CVs and job descriptions, retrieves relevant candidate experiences using RAG, generates match analysis, identifies skill gaps, recommends portfolio projects, and produces tailored cover letters and interview preparation outputs."

## What to Customize Before Pushing to GitHub

- Replace the sample data with your own polished demo examples
- Pick the OpenRouter model you want to showcase in `.env.example` or README notes
- Tighten prompts based on the types of jobs you want to target
- Add screenshots or a short GIF of the Streamlit app and FastAPI docs
- Pin dependency versions after your first successful local install
