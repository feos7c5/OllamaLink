import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, AsyncGenerator

logger = logging.getLogger(__name__)

class BaseClient(ABC):
    """
    Abstract base class for all provider clients.
    Defines the common interface and shared functionality.
    """
    
    def __init__(self, endpoint: str, name: str):
        self.endpoint = endpoint.rstrip('/')
        self.name = name
        self.available_models = []
        self.connection_error = None
        self.model_cache_time = 0
        self.cache_duration = 300  # 5 minutes default cache
        
        logger.info(f"{self.name} client initialized with endpoint: {self.endpoint}")
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the provider.
        Returns: {"status": "connected"|"error", "message": str}
        """
        pass
    
    @abstractmethod
    def fetch_models(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch available models from the provider.
        Returns: List of model dictionaries with at least 'id' and 'name' fields
        """
        pass
    
    @abstractmethod
    def get_model_name(self, requested_model: str, model_mappings: Dict[str, str] = None) -> str:
        """
        Map requested model to actual provider model name.
        Args:
            requested_model: The model name requested by the user
            model_mappings: Optional model mapping dictionary
        Returns: Actual model name to use with the provider
        """
        pass
    
    @abstractmethod
    def process_messages(self, messages: List[Dict[str, Any]], thinking_mode: bool = True) -> List[Dict[str, Any]]:
        """
        Process and format messages for the specific provider format.
        Args:
            messages: List of message dictionaries
            thinking_mode: Whether thinking mode is enabled
        Returns: Processed messages list
        """
        pass
    
    @abstractmethod
    async def chat_completion(self, model: str, messages: List[Dict[str, Any]], 
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            stream: bool = False) -> Dict[str, Any]:
        """
        Make a chat completion request.
        Args:
            model: Model name to use
            messages: List of message dictionaries
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
        Returns: Response dictionary with status and data/error
        """
        pass
    
    @abstractmethod
    async def stream_chat_completion(self, model: str, messages: List[Dict[str, Any]],
                                   temperature: float = 0.7, max_tokens: Optional[int] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a chat completion request.
        Args:
            model: Model name to use
            messages: List of message dictionaries
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
        Yields: Response chunks as dictionaries
        """
        pass
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model details by ID.
        Args:
            model_id: The model ID to search for
        Returns: Model dictionary or None if not found
        """
        for model in self.available_models:
            if model.get("id") == model_id:
                return model
        return None
    
    def search_models(self, query: str) -> List[Dict[str, Any]]:
        """
        Search models by name or description.
        Args:
            query: Search query string
        Returns: List of matching model dictionaries
        """
        query_lower = query.lower()
        matching_models = []
        
        for model in self.available_models:
            model_id = model.get("id", "").lower()
            model_name = model.get("name", "").lower()
            
            if (query_lower in model_id or 
                query_lower in model_name):
                matching_models.append(model)
        
        return matching_models
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models in OpenAI format.
        Returns: List of model dictionaries in OpenAI format
        """
        models_list = []
        
        for model in self.available_models:
            models_list.append({
                "id": model.get("id", model.get("name", "unknown")),
                "object": "model",
                "created": int(time.time()),
                "owned_by": self.name.lower(),
                "provider": self.name.lower()
            })
        
        return models_list
    
    def _is_cache_valid(self) -> bool:
        """Check if the model cache is still valid."""
        return (time.time() - self.model_cache_time) < self.cache_duration
    
    def _update_cache_time(self):
        """Update the cache timestamp."""
        self.model_cache_time = time.time()
    
    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name for consistent comparison."""
        # Remove version tags and normalize to lowercase
        normalized = name.lower()
        # Remove common suffixes
        for suffix in [':latest', ':v1', ':v2', ':instruct', ':chat']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                break
        return normalized
    
    def _extract_model_info(self, raw_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and normalize model information from provider-specific format.
        Args:
            raw_model: Raw model data from provider
        Returns: Normalized model dictionary
        """
        # Default implementation - subclasses can override for provider-specific fields
        return {
            "id": raw_model.get("id", raw_model.get("name", "unknown")),
            "name": raw_model.get("name", raw_model.get("id", "unknown")),
            "owned_by": self.name.lower(),
            "provider": self.name.lower()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of this client.
        Returns: Health status dictionary
        """
        return {
            "provider": self.name.lower(),
            "endpoint": self.endpoint,
            "models_available": len(self.available_models),
            "connection_error": self.connection_error,
            "cache_age_seconds": time.time() - self.model_cache_time if self.model_cache_time > 0 else None,
            "healthy": self.connection_error is None
        }
