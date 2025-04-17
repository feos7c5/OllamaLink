"""
OllamaLink Core Module - Bridge between OpenAI API and Ollama
"""

__version__ = "0.3.0"

from .api import create_api
from .request import OllamaRequestHandler
from .response import OllamaResponseHandler
from .router import OllamaRouter
from .util import is_valid_url, load_config, estimate_tokens, estimate_message_tokens
