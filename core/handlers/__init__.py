"""
Generic request and response handlers for all AI providers
"""

from .base_request_handler import BaseRequestHandler
from .base_response_handler import BaseResponseHandler
from .ollama_handlers import OllamaRequestHandler, OllamaResponseHandler
from .openrouter_handlers import OpenRouterRequestHandler, OpenRouterResponseHandler
from .llamacpp_handlers import LlamaCppRequestHandler, LlamaCppResponseHandler

__all__ = [
    'BaseRequestHandler', 
    'BaseResponseHandler',
    'OllamaRequestHandler',
    'OllamaResponseHandler', 
    'OpenRouterRequestHandler',
    'OpenRouterResponseHandler',
    'LlamaCppRequestHandler',
    'LlamaCppResponseHandler'
]
