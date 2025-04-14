"""
OllamaLink - A connector for Ollama models in Cursor AI

This package provides a bridge that allows Cursor AI to communicate with local Ollama models.
"""

__version__ = "0.1.0"

from .router import OllamaRouter
from .request import OllamaRequestHandler
from .response import OllamaResponseHandler
