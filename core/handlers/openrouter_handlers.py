import logging
import json
from typing import Dict, Any
import httpx
from .base_request_handler import BaseRequestHandler
from .base_response_handler import BaseResponseHandler

logger = logging.getLogger(__name__)


class OpenRouterRequestHandler(BaseRequestHandler):
    """OpenRouter-specific request handler implementation."""
    
    def __init__(self, endpoint: str, api_key: str, max_retries: int = 3, 
                 max_tokens_per_chunk: int = 8000, chunk_overlap: int = 1,
                 max_streaming_tokens: int = 32000):
        super().__init__(endpoint, max_retries, max_tokens_per_chunk, chunk_overlap, max_streaming_tokens)
        self.api_key = api_key
    
    def get_chat_url(self) -> str:
        """Return the OpenRouter chat completion URL."""
        return f"{self.endpoint}/api/v1/chat/completions"
    
    def get_health_url(self) -> str:
        """Return the OpenRouter health check URL."""
        return f"{self.endpoint}/api/v1/models"
    
    def prepare_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request data to OpenRouter format."""
        # OpenRouter uses OpenAI format directly
        prepared = request_data.copy()
        
        # Ensure stream is set properly
        if "stream" not in prepared:
            prepared["stream"] = self.prefer_streaming
        
        return prepared
    
    def get_request_headers(self, is_streaming: bool = False) -> Dict[str, str]:
        """Get OpenRouter-specific headers."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://ollamalink.local",
            "X-Title": "OllamaLink"
        }
        
        if is_streaming:
            headers.update({
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            })
        
        return headers
    
    def handle_error_response(self, response: httpx.Response, model: str) -> Dict[str, Any]:
        """Handle OpenRouter-specific error responses."""
        error_msg = f"OpenRouter returned status: {response.status_code}"
        
        try:
            error_data = response.json()
            if "error" in error_data:
                error_info = error_data["error"]
                
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", str(error_info))
                    error_code = error_info.get("code", response.status_code)
                else:
                    error_message = str(error_info)
                    error_code = response.status_code
                
                # Handle common OpenRouter errors
                if response.status_code == 401:
                    return {"error": {"message": "Invalid API key", "code": 401}}
                elif response.status_code == 402:
                    return {"error": {"message": "Insufficient credits", "code": 402}}
                elif response.status_code == 429:
                    return {"error": {"message": "Rate limit exceeded", "code": 429}}
                elif "model not found" in error_message.lower():
                    return {"error": {"message": f"Model '{model}' not available on OpenRouter", "code": 404}}
                else:
                    return {"error": {"message": error_message, "code": error_code}}
                    
        except Exception as parse_error:
            logger.error(f"Failed to parse OpenRouter error response: {parse_error}")
        
        return {"error": {"message": error_msg, "code": response.status_code}}


class OpenRouterResponseHandler(BaseResponseHandler):
    """OpenRouter-specific response handler implementation."""
    
    def parse_provider_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse OpenRouter response format."""
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenRouter response: {str(e)}")
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "I couldn't process your request. Please try again."
                    }
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0
                }
            }
    
    def extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from OpenRouter response."""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        return ""
    
    def extract_token_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from OpenRouter response."""
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
        }
    
    def parse_streaming_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenRouter streaming chunk."""
        parsed = {"content": ""}
        
        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            choice = chunk_data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                parsed["content"] = choice["delta"]["content"]
        
        return parsed
    
    def is_streaming_done(self, chunk_data: Dict[str, Any]) -> bool:
        """Check if OpenRouter streaming is complete."""
        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            choice = chunk_data["choices"][0]
            return choice.get("finish_reason") is not None
        return False
