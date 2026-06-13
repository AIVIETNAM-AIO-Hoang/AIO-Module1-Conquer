import requests
from unittest.mock import MagicMock, patch

from app.services.embedding import get_embedding


def test_get_embedding_returns_float_list():
    """get_embedding returns a non-empty list of floats on success."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3] * 256}
    mock_response.raise_for_status = MagicMock()

    with patch("app.services.embedding.requests.post", return_value=mock_response):
        result = get_embedding("hello world")

    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)


def test_get_embedding_raises_on_http_error():
    """get_embedding propagates HTTPError raised by raise_for_status."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

    with patch("app.services.embedding.requests.post", return_value=mock_response):
        try:
            get_embedding("hello")
            assert False, "Expected HTTPError"
        except requests.HTTPError:
            pass
