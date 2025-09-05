import logging
import httpx
import json
import requests
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from .base_client import BaseClient
from ..handlers import OllamaRequestHandler, OllamaResponseHandler

logger = logging.getLogger(__name__)

class OllamaClient(BaseClient):
    """
    Client for Ollama API integration.
    Handles model discovery, chat completions, and streaming for Ollama.
    """
    
    def __init__(self, endpoint: str = "http://localhost:11434"):
        super().__init__(endpoint, "Ollama")

        # Initialize handlers
        self.request_handler = OllamaRequestHandler(endpoint)
        self.response_handler = OllamaResponseHandler()

    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Ollama API."""
        try:
            response = requests.get(f"{self.endpoint}/api/version", timeout=10)
            
            if response.status_code == 200:
                version_data = response.json()
                return {
                    "status": "connected", 
                    "message": "Ollama connection successful",
                    "version": version_data.get("version", "unknown")
                }
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            logger.error(f"Ollama connection test failed: {str(e)}")
            return {"status": "error", "message": f"Connection failed: {str(e)}"}
    
    def fetch_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch available models from Ollama."""
        # Use cache if available and not expired
        if not force_refresh and self.available_models and self._is_cache_valid():
            logger.info(f"Using cached Ollama models ({len(self.available_models)} models)")
            return self.available_models
        
        self.connection_error = None
        
        try:
            logger.info(f"Fetching models from Ollama at {self.endpoint}")
            response = requests.get(f"{self.endpoint}/api/tags", timeout=15)
            logger.debug(f"Ollama response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Ollama response data: {data}")
                if "models" in data and len(data["models"]) > 0:
                    raw_models = data["models"]
                    
                    # Format models for consistency
                    formatted_models = []
                    for model in raw_models:
                        if model.get("name"):
                            formatted_models.append({
                                "id": model["name"],
                                "name": model.get("name", model["name"]),
                                "size": model.get("size", 0),
                                "modified_at": model.get("modified_at"),
                                "digest": model.get("digest"),
                                "details": model.get("details", {}),
                                "owned_by": "ollama"
                            })
                    
                    self.available_models = formatted_models
                    self._update_cache_time()
                    logger.info(f"Fetched {len(formatted_models)} models from Ollama")
                    return formatted_models
                else:
                    self.available_models = []
                    logger.warning("No models found in Ollama")
                    return []
            else:
                self.connection_error = f"HTTP {response.status_code}"
                logger.error(f"Failed to fetch Ollama models: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.connection_error = str(e)
            logger.error(f"Error fetching Ollama models: {str(e)}")
            self.available_models = []
            return []
    
    def get_model_name(self, requested_model: str, model_mappings: Dict[str, str] = None) -> str:
        """Map requested model to available Ollama model."""
        if not self.available_models:
            logger.warning("No Ollama models available")
            return model_mappings.get("default", "llama3") if model_mappings else "llama3"
        
        model_mappings = model_mappings or {}
        
        # Check if model is in mappings
        if requested_model in model_mappings:
            mapped_model = model_mappings[requested_model]
            # Try exact match first
            if any(m["id"] == mapped_model for m in self.available_models):
                return mapped_model
            # Try fuzzy match
            normalized_mapped = self._normalize_model_name(mapped_model)
            for model in self.available_models:
                if self._normalize_model_name(model["id"]) == normalized_mapped:
                    return model["id"]
        
        # Try direct match
        if any(m["id"] == requested_model for m in self.available_models):
            return requested_model
        
        # Try fuzzy matching
        normalized_requested = self._normalize_model_name(requested_model)
        for model in self.available_models:
            normalized_available = self._normalize_model_name(model["id"])
            if normalized_available == normalized_requested or normalized_available.startswith(normalized_requested):
                return model["id"]
        
        # Use default
        default_model = (model_mappings.get("default") if model_mappings 
                        else self.available_models[0]["id"] if self.available_models else "llama3")
        logger.warning(f"No match found for {requested_model}, using default: {default_model}")
        return default_model
    
    def process_messages(self, messages: List[Dict[str, Any]], thinking_mode: bool = True) -> List[Dict[str, Any]]:
        """Process messages based on thinking mode setting."""
        if thinking_mode:
            return messages
            
        processed_messages = []
        for message in messages:
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                
                if isinstance(content, str) and not content.startswith("/no_think"):
                    message = message.copy()
                    message["content"] = f"/no_think {content}"
                    logger.info(f"Added /no_think prefix to user message (thinking mode disabled)")
                    
            processed_messages.append(message)
            
        return processed_messages
    
    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], 
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            stream: bool = False) -> Dict[str, Any]:
        """Make a chat completion request to Ollama."""
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            request_data["max_tokens"] = max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    f"{self.endpoint}/api/chat",
                    json=request_data
                )
                
                if response.status_code == 200:
                    if stream:
                        return {"status": "streaming", "response": response}
                    else:
                        return {"status": "success", "data": response.json()}
                else:
                    return {
                        "status": "error",
                        "code": response.status_code,
                        "error": {"message": f"Ollama returned status: {response.status_code}"}
                    }
                    
        except Exception as e:
            logger.error(f"Ollama chat completion error: {str(e)}")
            return {
                "status": "error",
                "error": {"message": f"Request failed: {str(e)}"}
            }
    
    async def stream_chat_completion(self, model: str, messages: List[Dict[str, Any]],
                                   temperature: float = 0.7, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion from Ollama."""
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        
        if max_tokens:
            request_data["max_tokens"] = max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.endpoint}/api/chat",
                    json=request_data
                ) as response:
                    
                    if response.status_code != 200:
                        yield {
                            "error": {"message": f"Ollama returned status: {response.status_code}", "code": response.status_code}
                        }
                        return
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama streaming error: {str(e)}")
            yield {
                "error": {"message": f"Streaming failed: {str(e)}"}
            }
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model details by ID."""
        for model in self.available_models:
            if model["id"] == model_id:
                return model
        return None
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """Search models by name."""
        query_lower = query.lower()
        matching_models = []
        
        for model in self.available_models:
            if (query_lower in model["id"].lower() or 
                query_lower in model.get("name", "").lower()):
                matching_models.append(model)
        
        return matching_models
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models in OpenAI format."""
        models_list = []
        
        for model in self.available_models:
            models_list.append({
                "id": model["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "provider": "ollama"
            })
        
        return models_list

