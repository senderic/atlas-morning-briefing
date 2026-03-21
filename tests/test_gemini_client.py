import pytest
from unittest.mock import patch, MagicMock
from scripts.gemini_client import GeminiClient

@patch('scripts.gemini_client.os.environ.get')
def test_gemini_client_initialization(mock_env_get):
    mock_env_get.return_value = "dummy_api_key"
    client = GeminiClient()
    assert client.api_key == "dummy_api_key"
    assert client.enabled is True
    assert client.models["heavy"] == "gemini-2.5-pro"

@patch('scripts.gemini_client.os.environ.get')
@patch('scripts.gemini_client.genai')
def test_gemini_client_invoke_success(mock_genai, mock_env_get):
    mock_env_get.return_value = "dummy_api_key"
    client = GeminiClient()
    client._available = True

    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello Gemini"
    mock_model.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model

    result = client.invoke("Test prompt")
    assert result == "Hello Gemini"
