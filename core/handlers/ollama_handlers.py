import logging
import json
from typing import Dict, Any
import httpx
from .base_request_handler import BaseRequestHandler
from .base_response_handler import BaseResponseHandler

logger = logging.getLogger(__name__)


class OllamaRequestHandler(BaseRequestHandler):
    """Ollama-specific request handler implementation."""
    
    def get_chat_url(self) -> str:
        """Return the Ollama chat completion URL."""
        return f"{self.endpoint}/api/chat"
    
    def get_health_url(self) -> str:
        """Return the Ollama health check URL."""
        return f"{self.endpoint}/api/version"
    
    def prepare_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request data to Ollama format."""
        # Ollama uses the same format as OpenAI, so minimal transformation needed
        prepared = request_data.copy()
        
        # Ensure stream is set properly
        if "stream" not in prepared:
            prepared["stream"] = self.prefer_streaming
        
        return prepared
    
    def get_request_headers(self, is_streaming: bool = False) -> Dict[str, str]:
        """Get Ollama-specific headers."""
        headers = {
            "Content-Type": "application/json"
        }
        
        if is_streaming:
            headers.update({
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            })
        
        return headers
    
    def handle_error_response(self, response: httpx.Response, model: str) -> Dict[str, Any]:
        """Handle Ollama-specific error responses."""
        error_msg = f"Ollama returned status: {response.status_code}"
        
        try:
            error_data = response.json()
            if "error" in error_data:
                error_text = error_data["error"].lower()
                
                if "model not found" in error_text:
                    return {"error": {"message": f"Model '{model}' not found", "code": 404}}
                elif "context length" in error_text or "too long" in error_text:
                    return {"error": {"message": "Context length exceeded", "code": 413}}
                elif "no such file" in error_text:
                    return {"error": {"message": f"Model '{model}' not available", "code": 404}}
                else:
                    return {"error": {"message": error_data["error"], "code": response.status_code}}
        except:
            pass
        
        return {"error": {"message": error_msg, "code": response.status_code}}


class OllamaResponseHandler(BaseResponseHandler):
    """Ollama-specific response handler implementation."""
    
    def parse_provider_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse Ollama response format."""
        try:
            return response.json()
        except json.JSONDecodeError:
            try:
                content = response.text.strip()
                if "\n" in content:
                    # Handle multi-line JSON (take first valid line)
                    first_json = content.split("\n")[0].strip()
                    return json.loads(first_json)
                else:
                    return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to parse Ollama response: {str(e)}")
                return {
                    "message": {
                        "role": "assistant", 
                        "content": "I couldn't process your request. Please try again."
                    },
                    "prompt_eval_count": 0,
                    "eval_count": 0
                }
    
    def extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract content from Ollama response."""
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        return ""
    
    def extract_token_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from Ollama response."""
        return {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0)
        }
    
    def parse_streaming_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Ollama streaming chunk."""
        parsed = {"content": ""}
        
        if "message" in chunk_data and "content" in chunk_data["message"]:
            parsed["content"] = chunk_data["message"]["content"]
        
        return parsed
    
    def is_streaming_done(self, chunk_data: Dict[str, Any]) -> bool:
        """Check if Ollama streaming is complete."""
        return chunk_data.get("done", False)
