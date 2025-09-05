import logging
import json
from typing import Dict, Any
import httpx
from .base_request_handler import BaseRequestHandler
from .base_response_handler import BaseResponseHandler

logger = logging.getLogger(__name__)


class LlamaCppRequestHandler(BaseRequestHandler):
    """Llama.cpp-specific request handler implementation."""
    
    def get_chat_url(self) -> str:
        """Return the Llama.cpp chat completion URL."""
        return f"{self.endpoint}/v1/chat/completions"
    
    def get_health_url(self) -> str:
        """Return the Llama.cpp health check URL."""
        return f"{self.endpoint}/v1/models"
    
    def prepare_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request data to Llama.cpp format."""
        # Llama.cpp uses OpenAI-compatible format
        prepared = request_data.copy()
        
        # Ensure stream is set properly
        if "stream" not in prepared:
            prepared["stream"] = self.prefer_streaming
        
        # Llama.cpp often has a default model, so we can set it if not provided
        if "model" not in prepared or not prepared["model"]:
            prepared["model"] = "default"
        
        return prepared
    
    def get_request_headers(self, is_streaming: bool = False) -> Dict[str, str]:
        """Get Llama.cpp-specific headers."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if is_streaming:
            headers.update({
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache"
            })
        
        return headers
    
    def handle_error_response(self, response: httpx.Response, model: str) -> Dict[str, Any]:
        """Handle Llama.cpp-specific error responses."""
        error_msg = f"Llama.cpp returned status: {response.status_code}"
        
        try:
            error_data = response.json()
            if "error" in error_data:
                error_info = error_data["error"]
                
                if isinstance(error_info, dict):
                    error_message = error_info.get("message", str(error_info))
                else:
                    error_message = str(error_info)
                
                # Handle common Llama.cpp errors
                if response.status_code == 404:
                    return {"error": {"message": f"Endpoint not found - check Llama.cpp server", "code": 404}}
                elif response.status_code == 400:
                    return {"error": {"message": f"Bad request: {error_message}", "code": 400}}
                elif "context" in error_message.lower() or "length" in error_message.lower():
                    return {"error": {"message": "Context length exceeded", "code": 413}}
                else:
                    return {"error": {"message": error_message, "code": response.status_code}}
                    
        except Exception as parse_error:
            logger.error(f"Failed to parse Llama.cpp error response: {parse_error}")
        
        # Handle specific status codes
        if response.status_code == 503:
            return {"error": {"message": "Llama.cpp server unavailable", "code": 503}}
        elif response.status_code == 500:
            return {"error": {"message": "Llama.cpp server error", "code": 500}}
        
        return {"error": {"message": error_msg, "code": response.status_code}}


class LlamaCppResponseHandler(BaseResponseHandler):
    """Llama.cpp-specific response handler implementation."""
    
    def parse_provider_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse Llama.cpp response format."""
        try:
            return response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Llama.cpp response: {str(e)}")
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
        """Extract content from Llama.cpp response."""
        if "choices" in response and len(response["choices"]) > 0:
            choice = response["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
        return ""
    
    def extract_token_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from Llama.cpp response."""
        usage = response.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0)
        }
    
    def parse_streaming_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Llama.cpp streaming chunk."""
        parsed = {"content": ""}
        
        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            choice = chunk_data["choices"][0]
            if "delta" in choice and "content" in choice["delta"]:
                parsed["content"] = choice["delta"]["content"]
        
        return parsed
    
    def is_streaming_done(self, chunk_data: Dict[str, Any]) -> bool:
        """Check if Llama.cpp streaming is complete."""
        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            choice = chunk_data["choices"][0]
            return choice.get("finish_reason") is not None
        return False
