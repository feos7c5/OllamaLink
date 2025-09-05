import logging
import time
import json
import httpx
import requests
from typing import Dict, List, Any, Optional, AsyncGenerator
from .base_client import BaseClient
from ..handlers import OpenRouterRequestHandler, OpenRouterResponseHandler

logger = logging.getLogger(__name__)

class OpenRouterClient(BaseClient):
    """
    Client for OpenRouter.ai API integration.
    Handles authentication, model discovery, chat completions, and streaming.
    """
    
    def __init__(self, api_key: str, endpoint: str = "https://openrouter.ai"):
        super().__init__(endpoint, "OpenRouter")
        self.api_key = api_key
        self.cache_duration = 600  # 10 minutes cache for OpenRouter models
        
        # Initialize handlers
        self.request_handler = OpenRouterRequestHandler(endpoint, api_key)
        self.response_handler = OpenRouterResponseHandler()
        
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/ollamalink",
            "X-Title": "OllamaLink"
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to OpenRouter API."""
        try:
            response = requests.get(
                f"{self.endpoint}/api/v1/models",
                headers=self._get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "status": "connected",
                    "message": "OpenRouter connection successful"
                }
            elif response.status_code == 401:
                return {
                    "status": "error",
                    "message": "Invalid API key"
                }
            elif response.status_code == 402:
                return {
                    "status": "error", 
                    "message": "Insufficient credits"
                }
            else:
                return {
                    "status": "error",
                    "message": f"OpenRouter returned status {response.status_code}"
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "message": "Cannot connect to OpenRouter server"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            }
    
    async def fetch_models(self) -> Dict[str, Any]:
        """Fetch available models from OpenRouter."""
        logger.info("Fetching models from OpenRouter")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.endpoint}/api/v1/models",
                    headers=self._get_headers(),
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    
                    model_list = []
                    for model in models:
                        model_id = model.get("id", "unknown")
                        model_list.append({
                            "id": model_id,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": model.get("owned_by", "openrouter"),
                            "context_length": model.get("context_length", 0),
                            "pricing": model.get("pricing", {})
                        })
                    
                    self._model_cache = model_list
                    self._last_fetch = time.time()
                    
                    logger.info(f"Successfully fetched {len(model_list)} models from OpenRouter")
                    return {"models": model_list}
                else:
                    error_msg = f"Failed to fetch models: HTTP {response.status_code}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                    
        except Exception as e:
            error_msg = f"Error fetching OpenRouter models: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], 
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            stream: bool = False) -> Dict[str, Any]:
        """Handle chat completion request using the new handler architecture."""
        try:
            # Prepare request data
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }
            
            if max_tokens:
                request_data["max_tokens"] = max_tokens
            
            # Use the new request handler
            response = await self.request_handler.handle_chat_request(request_data)
            
            if isinstance(response, dict) and "error" in response:
                return response
            
            # Use the new response handler
            if stream:
                return self.response_handler.handle_response(response, model, is_streaming=True)
            else:
                return self.response_handler.handle_response(response, model, is_streaming=False)
                
        except Exception as e:
            logger.error(f"Chat completion error: {str(e)}")
            return {"error": {"message": f"Chat completion failed: {str(e)}", "code": 500}}

    async def stream_chat_completion(self, model: str, messages: List[Dict[str, Any]], 
                                   temperature: float = 0.7, max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        """Handle streaming chat completion using the new handler architecture."""
        try:
            # Prepare request data
            request_data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            
            if max_tokens:
                request_data["max_tokens"] = max_tokens
            
            # Use the new request handler
            response = await self.request_handler.handle_chat_request(request_data)
            
            logger.info(f"OpenRouter streaming response type: {type(response)}")
            
            if isinstance(response, dict) and "error" in response:
                # Yield error in SSE format
                logger.error(f"OpenRouter streaming error: {response['error']}")
                yield f"data: {json.dumps(response)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Check if response is an httpx.Response object
            if not hasattr(response, 'aiter_lines'):
                logger.error(f"Invalid response type for streaming: {type(response)}")
                error_response = {"error": {"message": "Invalid streaming response", "code": 500}}
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Use the new response handler for streaming
            async for chunk in self.response_handler.stream_response(response, model):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            error_response = {"error": {"message": f"Streaming failed: {str(e)}", "code": 500}}
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "data: [DONE]\n\n"

    def get_model_name(self, requested_model: str) -> str:
        """Get the actual model name to use for OpenRouter."""
        # Use the base client's model mapping functionality
        return self._resolve_model_name(requested_model)

    def process_messages(self, messages: List[Dict[str, Any]], thinking_mode: bool = True) -> List[Dict[str, Any]]:
        """Process messages for OpenRouter format."""
        # OpenRouter uses standard OpenAI format, minimal processing needed
        processed_messages = []
        
        for message in messages:
            # Handle structured content
            if isinstance(message.get("content"), list):
                content_parts = []
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        content_parts.append(item)
                
                processed_message = message.copy()
                processed_message["content"] = " ".join(content_parts)
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        
        return processed_messages
