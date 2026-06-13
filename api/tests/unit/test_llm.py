from unittest.mock import MagicMock, patch

from app.services.llm import generate_answer


def test_generate_answer_returns_non_empty_string():
    """generate_answer returns a non-empty string on success."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "The answer is 42."}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.llm.requests.post", return_value=mock_response):
        result = generate_answer("What is the answer?", ["The answer is 42."])

    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_answer_includes_context_in_prompt():
    """generate_answer embeds all context chunks in the outgoing prompt."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "answer"}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.llm.requests.post", return_value=mock_response) as mock_post:
        generate_answer("my question", ["chunk one", "chunk two"])

    payload = mock_post.call_args.kwargs["json"]
    assert "chunk one" in payload["prompt"]
    assert "chunk two" in payload["prompt"]
    assert "my question" in payload["prompt"]


def test_generate_answer_uses_configured_model():
    """generate_answer sends the model name from settings."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ok"}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.llm.requests.post", return_value=mock_response) as mock_post:
        generate_answer("q", ["ctx"])

    from app.config import settings

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == settings.OLLAMA_LLM_MODEL
