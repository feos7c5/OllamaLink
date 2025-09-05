import logging
import asyncio
import httpx
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..util import estimate_message_tokens, count_tokens_in_messages

logger = logging.getLogger(__name__)


class BaseRequestHandler(ABC):
    """
    Abstract base class for handling HTTP requests to AI providers.
    
    This class provides common functionality like retries, timeouts, token management,
    and error handling that can be used by all provider clients.
    """
    
    def __init__(self, endpoint: str, max_retries: int = 3, 
                 max_tokens_per_chunk: int = 8000, chunk_overlap: int = 1,
                 max_streaming_tokens: int = 32000):
        self.endpoint = endpoint
        self.max_retries = max_retries
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.prefer_streaming = True
        self.max_streaming_tokens = max_streaming_tokens
        self._client = None
    
    def __del__(self):
        """Cleanup when the handler is destroyed."""
        if self._client is not None and not self._client.is_closed:
            try:
                import asyncio
                # Try to close the client if there's an active event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.close_client())
            except:
                pass
    
    @abstractmethod
    def get_chat_url(self) -> str:
        """Return the chat completion URL for this provider."""
        pass
    
    @abstractmethod
    def get_health_url(self) -> str:
        """Return the health check URL for this provider."""
        pass
    
    @abstractmethod
    def prepare_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform request data to provider-specific format."""
        pass
    
    @abstractmethod
    def get_request_headers(self, is_streaming: bool = False) -> Dict[str, str]:
        """Get provider-specific headers for requests."""
        pass
    
    @abstractmethod
    def handle_error_response(self, response: httpx.Response, model: str) -> Dict[str, Any]:
        """Handle provider-specific error responses."""
        pass
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=5.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client
    
    async def close_client(self):
        """Close the HTTP client if it exists."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def test_connection(self) -> bool:
        """Test connection to the provider."""
        try:
            client = await self.get_client()
            health_response = await client.get(self.get_health_url())
            return health_response.status_code == 200
        except Exception:
            return False
    
    def sanitize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize message content, handling complex content structures."""
        sanitized_messages = []
        for msg in messages:
            sanitized_msg = msg.copy()
            if "content" in sanitized_msg and isinstance(sanitized_msg["content"], list):
                content_parts = []
                for item in sanitized_msg["content"]:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            content_parts.append(item["text"])
                    elif isinstance(item, str):
                        content_parts.append(item)
                sanitized_msg["content"] = " ".join(content_parts)
            sanitized_messages.append(sanitized_msg)
        return sanitized_messages
    
    async def make_request(self, request_data: Dict[str, Any], 
                          timeout_seconds: int = 90) -> httpx.Response:
        """
        Make HTTP request to provider with retries and error handling.
        """
        url = self.get_chat_url()
        retry_count = 0
        model = request_data.get("model", "unknown")
        
        # Sanitize messages
        if "messages" in request_data:
            request_data["messages"] = self.sanitize_messages(request_data["messages"])
        
        # Prepare provider-specific request data
        prepared_data = self.prepare_request_data(request_data)
        
        is_streaming = prepared_data.get("stream", self.prefer_streaming)
        
        # Test connection first
        if not await self.test_connection():
            return {"error": {"message": "Cannot connect to provider API", "code": 503}}
        
        while retry_count < self.max_retries:
            try:
                client = await self.get_client()
                logger.info(f"Requesting model: {model} (retry {retry_count+1}/{self.max_retries}) streaming={is_streaming}")
                
                headers = self.get_request_headers(is_streaming)
                
                # Update timeout for non-streaming requests
                if not is_streaming:
                    client._timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=5.0)
                
                response = await client.post(url, json=prepared_data, headers=headers)

                if response.status_code == 200:
                    return response
                else:
                    return self.handle_error_response(response, model)
                        
            except httpx.TimeoutException:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Request timeout after {self.max_retries} retries")
                    return {"error": {"message": "Request timeout", "code": 408}}
                await asyncio.sleep(min(2 ** retry_count, 8))
                
            except Exception as e:
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Request failed after {self.max_retries} retries: {str(e)}")
                    return {"error": {"message": f"Request failed: {str(e)}", "code": 500}}
                await asyncio.sleep(min(2 ** retry_count, 8))
        
        return {"error": {"message": "Max retries exceeded", "code": 500}}
    
    def chunk_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split messages into chunks based on token limits.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for message in messages:
            message_tokens = estimate_message_tokens(message)
            
            if current_tokens + message_tokens > self.max_tokens_per_chunk and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(message)
            current_tokens += message_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    async def process_chunked_request(self, original_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process large requests by chunking messages.
        """
        messages = original_request.get("messages", [])
        chunks = self.chunk_messages(messages)
        
        if len(chunks) <= 1:
            return await self.make_request(original_request)
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        current_context = []
        
        for i, chunk in enumerate(chunks):
            chunk_request = original_request.copy()
            chunk_request["messages"] = current_context + chunk
            chunk_request["stream"] = False  # Force non-streaming for chunks
            
            response = await self.make_request(chunk_request)
            
            if isinstance(response, dict) and "error" in response:
                return response
            
            # This would need to be implemented per provider
            # For now, return the last response
            if i == len(chunks) - 1:
                return response
            
            # Update context (provider-specific logic needed)
            # current_context = self._update_context(response, current_context)
            
        return {"error": {"message": "Chunking processing error", "code": 500}}
    
    async def process_large_streaming_request(self, request_data: Dict[str, Any]) -> httpx.Response:
        """
        Process large streaming requests by reducing context.
        """
        messages = request_data.get("messages", [])
        total_tokens = count_tokens_in_messages(messages)
        logger.info(f"Processing large streaming request ({len(messages)} messages, ~{total_tokens} tokens)")
        
        max_safe_tokens = min(self.max_streaming_tokens, 12000)
        
        if total_tokens <= max_safe_tokens:
            logger.info("Request within token limit")
            return await self.make_request(request_data, min(request_data.get("timeout", 90), 90))
            
        # Reduce context by keeping system messages and recent messages
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        available_tokens = max_safe_tokens - system_tokens - 200
        
        preserved_messages = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = estimate_message_tokens(msg)
            if current_tokens + msg_tokens <= available_tokens:
                preserved_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        reduced_messages = system_messages + preserved_messages
        
        if len(reduced_messages) < len(messages):
            logger.info(f"Reduced context from {len(messages)} to {len(reduced_messages)} messages")
            
            context_note = {
                "role": "system",
                "content": f"Note: Only the most recent {len(preserved_messages)} messages were included due to token limits."
            }
            reduced_messages.insert(len(system_messages), context_note)
        
        modified_request = request_data.copy()
        modified_request["messages"] = reduced_messages
        
        timeout = min(request_data.get("timeout", 180), 180)
        
        return await self.make_request(modified_request, timeout)
    
    async def handle_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for handling chat completion requests.
        """
        timeout = request_data.get("timeout", 180)
        request_data["timeout"] = timeout
        
        message_count = len(request_data.get("messages", []))
        token_count = count_tokens_in_messages(request_data.get("messages", []))
        is_large_request = message_count > 5 or token_count > 6000
        
        is_stream = request_data.get("stream", True)
        request_data["stream"] = is_stream
        
        logger.info(f"Request: {message_count} msgs, ~{token_count} tokens, streaming={is_stream}")
        
        try:
            if not is_large_request:
                return await self.make_request(request_data)
            elif is_stream:
                return await self.process_large_streaming_request(request_data)
            else:
                return await self.process_chunked_request(request_data)
                
        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            return {"error": {"message": "Error processing request", "code": 500}}
    
    def calculate_request_size(self, request_data: Dict[str, Any]) -> int:
        """Calculate the token size of a request."""
        return count_tokens_in_messages(request_data.get("messages", []))
