from pathlib import Path

from app.document_loader import load_cv_text_from_path


def test_text_cv_loading(tmp_path: Path) -> None:
    cv_path = tmp_path / "candidate_cv.txt"
    cv_path.write_text("Python developer\nBuilt a chatbot with FastAPI.\n", encoding="utf-8")

    cv_text = load_cv_text_from_path(cv_path)

    assert "Python developer" in cv_text
    assert "FastAPI" in cv_text

