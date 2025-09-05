import logging
import httpx
import json
import requests
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from .base_client import BaseClient
from ..handlers import LlamaCppRequestHandler, LlamaCppResponseHandler

logger = logging.getLogger(__name__)

class LlamaCppClient(BaseClient):
    """
    Client for llama.cpp server API integration.
    Handles model discovery, chat completions, and streaming for llama.cpp.
    """
    
    def __init__(self, endpoint: str = "http://localhost:8080"):
        super().__init__(endpoint, "LlamaCpp")
        self.current_model = None  # llama.cpp typically loads one model at a time
        
        # Initialize handlers
        self.request_handler = LlamaCppRequestHandler(endpoint)
        self.response_handler = LlamaCppResponseHandler()
        
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to llama.cpp server."""
        try:
            # Check if llama.cpp server is running
            response = requests.get(f"{self.endpoint}/health", timeout=10)
            
            if response.status_code == 200:
                # Get model info
                try:
                    props_response = requests.get(f"{self.endpoint}/props", timeout=10)
                    if props_response.status_code == 200:
                        props = props_response.json()
                        model_name = props.get("default_generation_settings", {}).get("model", "unknown")
                        self.current_model = model_name
                        
                        return {
                            "status": "connected",
                            "message": "Llama.cpp connection successful",
                            "model": model_name
                        }
                except:
                    pass
                
                return {
                    "status": "connected",
                    "message": "Llama.cpp connection successful"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Llama.cpp returned status {response.status_code}"
                }
        except requests.exceptions.ConnectionError:
            return {
                "status": "error",
                "message": "Cannot connect to Llama.cpp server"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Connection test failed: {str(e)}"
            }

    async def fetch_models(self) -> Dict[str, Any]:
        """Fetch available models from llama.cpp server."""
        logger.info(f"Fetching models from Llama.cpp at {self.endpoint}")
        
        try:
            async with httpx.AsyncClient() as client:
                # Try v1/models endpoint first (OpenAI-compatible)
                response = await client.get(f"{self.endpoint}/v1/models", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("data", [])
                    
                    if models:
                        model_list = []
                        for model in models:
                            model_id = model.get("id", "default")
                            model_list.append({
                                "id": model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "llamacpp"
                            })
                        
                        self._model_cache = model_list
                        self._last_fetch = time.time()
                        
                        logger.info(f"Successfully fetched {len(model_list)} models from Llama.cpp")
                        return {"models": model_list}
                
                # Fallback: create a default model entry
                default_model = [{
                    "id": self.current_model or "default",
                    "object": "model", 
                    "created": int(time.time()),
                    "owned_by": "llamacpp"
                }]
                
                self._model_cache = default_model
                self._last_fetch = time.time()
                
                logger.info("Using default model for Llama.cpp")
                return {"models": default_model}
                    
        except Exception as e:
            error_msg = f"Error fetching Llama.cpp models: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}

    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], 
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            stream: bool = False) -> Dict[str, Any]:
        """Handle chat completion request using the new handler architecture."""
        try:
            # Prepare request data
            request_data = {
                "model": model or "default",
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
                "model": model or "default",
                "messages": messages,
                "temperature": temperature,
                "stream": True
            }
            
            if max_tokens:
                request_data["max_tokens"] = max_tokens
            
            # Use the new request handler
            response = await self.request_handler.handle_chat_request(request_data)
            
            if isinstance(response, dict) and "error" in response:
                # Yield error in SSE format
                yield f"data: {json.dumps(response)}\n\n"
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
        """Get the actual model name to use for Llama.cpp."""
        # Use the base client's model mapping functionality
        return self._resolve_model_name(requested_model)

    def process_messages(self, messages: List[Dict[str, Any]], thinking_mode: bool = True) -> List[Dict[str, Any]]:
        """Process messages for Llama.cpp format."""
        # Llama.cpp expects messages in OpenAI format, so minimal processing needed
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
