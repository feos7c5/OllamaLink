import logging
import json
import time
import random
import asyncio
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator
import httpx

logger = logging.getLogger(__name__)


class BaseResponseHandler(ABC):
    """
    Abstract base class for handling HTTP responses from AI providers.
    
    This class provides common functionality for parsing responses, converting to OpenAI format,
    and handling streaming responses that can be used by all provider clients.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def parse_provider_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse provider-specific response format."""
        pass
    
    @abstractmethod
    def extract_content_from_response(self, response: Dict[str, Any]) -> str:
        """Extract the assistant's content from provider response."""
        pass
    
    @abstractmethod
    def extract_token_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage information from provider response."""
        pass
    
    @abstractmethod
    def parse_streaming_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single streaming chunk from provider."""
        pass
    
    @abstractmethod
    def is_streaming_done(self, chunk_data: Dict[str, Any]) -> bool:
        """Check if streaming is complete based on chunk data."""
        pass
    
    def generate_message_id(self) -> str:
        """Generate a unique message ID for OpenAI compatibility."""
        return f"chatcmpl-{random.randint(10000, 99999)}"
    
    def get_current_timestamp(self) -> int:
        """Get current Unix timestamp."""
        return int(time.time())
    
    def format_openai_response(self, response: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        """Format provider response as OpenAI-compatible response."""
        assistant_content = self.extract_content_from_response(response)
        token_usage = self.extract_token_usage(response)
        
        return {
            "id": self.generate_message_id(),
            "object": "chat.completion",
            "created": self.get_current_timestamp(),
            "model": requested_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("prompt_tokens", 0) + token_usage.get("completion_tokens", 0)
            },
            "system_fingerprint": "ollamalink-server"
        }
    
    def format_openai_error(self, error_message: str, error_code: int = 500) -> Dict[str, Any]:
        """Format error as OpenAI-compatible error response."""
        return {
            "error": {
                "message": error_message,
                "type": "api_error",
                "code": error_code
            }
        }
    
    def create_streaming_chunk(self, content: str, message_id: str, model: str, 
                              finish_reason: str = None) -> Dict[str, Any]:
        """Create OpenAI-compatible streaming chunk."""
        chunk = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": self.get_current_timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        }
        
        if content:
            chunk["choices"][0]["delta"]["content"] = content
        
        return chunk
    
    def create_role_chunk(self, message_id: str, model: str) -> Dict[str, Any]:
        """Create the initial role chunk for streaming."""
        return {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": self.get_current_timestamp(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant"
                },
                "finish_reason": None
            }]
        }
    
    async def stream_response(self, response: httpx.Response, 
                            requested_model: str) -> AsyncGenerator[str, None]:
        """Process streaming response from provider into OpenAI format."""
        try:
            logger.info(f"Starting stream processing for {requested_model}")
            chunk_index = 0
            message_id = self.generate_message_id()
            created_time = self.get_current_timestamp()
            last_yield_time = time.time()
            start_time = time.time()
            
            # Send initial role chunk
            role_event = self.create_role_chunk(message_id, requested_model)
            logger.debug(f"Sending role chunk: {role_event}")
            yield f"data: {json.dumps(role_event)}\n\n"
            
            keepalive_interval = 5
            max_idle_time = 60  # Only timeout if NO data received for 60 seconds
            lines_processed = 0
            last_data_time = time.time()
            
            logger.info(f"Starting to iterate over response lines...")
            async for line in response.aiter_lines():
                current_time = time.time()
                elapsed_time = current_time - start_time
                idle_time = current_time - last_data_time
                
                lines_processed += 1
                logger.debug(f"Received line {lines_processed}: {repr(line)}")
                
                # Update last data time when we receive actual content
                if line.strip():
                    last_data_time = current_time
                
                # Only timeout if we haven't received ANY data for max_idle_time
                if idle_time > max_idle_time:
                    logger.warning(f"Stream idle timeout after {idle_time:.1f}s with no data")
                    
                    timeout_chunk = self.create_streaming_chunk(
                        "\n\n[Connection lost - no data received]",
                        message_id, requested_model
                    )
                    yield f"data: {json.dumps(timeout_chunk)}\n\n"
                    
                    finish_chunk = self.create_streaming_chunk(
                        "", message_id, requested_model, "stop"
                    )
                    yield f"data: {json.dumps(finish_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                
                # Send keepalive
                if current_time - last_yield_time > keepalive_interval:
                    logger.debug("Sending keepalive")
                    yield f": keepalive {int(current_time)}\n\n"
                    last_yield_time = current_time
                
                if not line.strip():
                    logger.debug("Skipping empty line")
                    continue
                
                try:
                    # Handle SSE format
                    if line.startswith("data: "):
                        data_content = line[6:]  # Remove "data: " prefix
                        
                        if data_content.strip() == "[DONE]":
                            logger.debug("Received [DONE] marker")
                            break
                        
                        chunk_data = json.loads(data_content)
                    else:
                        # Handle direct JSON format
                        chunk_data = json.loads(line)
                    
                    # Handle errors
                    if "error" in chunk_data:
                        logger.error(f"Stream error: {chunk_data['error']}")
                        error_response = self.format_openai_error("Stream error", 500)
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    # Parse chunk using provider-specific logic
                    parsed_chunk = self.parse_streaming_chunk(chunk_data)
                    logger.debug(f"Parsed chunk: {parsed_chunk}")
                    
                    # Send content if available  
                    if parsed_chunk.get("content"):
                        content = parsed_chunk["content"]
                        logger.debug(f"Sending content: {repr(content)}")
                        
                        # Split content for smoother streaming experience
                        if len(content) > 1:
                            # Split by words and send each word separately
                            # Split on word boundaries, keeping punctuation with words
                            words = re.findall(r'\S+|\s+', content)
                            
                            for word in words:
                                # Don't skip spaces - they're important for readability!
                                content_chunk = self.create_streaming_chunk(
                                    word, message_id, requested_model
                                )
                                yield f"data: {json.dumps(content_chunk)}\n\n"
                                chunk_index += 1
                                
                                # Only add delay for actual words, not spaces
                                if word.strip():
                                    await asyncio.sleep(0.03)  # 30ms delay
                        else:
                            # Single character, send as-is
                            content_chunk = self.create_streaming_chunk(
                                content, message_id, requested_model
                            )
                            yield f"data: {json.dumps(content_chunk)}\n\n"
                            chunk_index += 1
                        
                        last_yield_time = current_time
                    
                    # Check if streaming is done
                    if self.is_streaming_done(chunk_data):
                        finish_chunk = self.create_streaming_chunk(
                            "", message_id, requested_model, "stop"
                        )
                        yield f"data: {json.dumps(finish_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    pass
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
            
            logger.info(f"Stream completed: {lines_processed} lines, {chunk_index} chunks in {time.time() - start_time:.2f}s")
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            try:
                error_response = self.format_openai_error("Stream error", 500)
                yield f"data: {json.dumps(error_response)}\n\n"
                yield "data: [DONE]\n\n"
            except:
                pass
    
    def handle_response(self, response: httpx.Response, requested_model: str, 
                       is_streaming: bool = False) -> Any:
        """
        Main entry point for handling responses.
        
        Args:
            response: HTTP response from provider
            requested_model: The model name requested by client
            is_streaming: Whether this is a streaming response
            
        Returns:
            For streaming: AsyncGenerator yielding SSE chunks
            For non-streaming: Dict with OpenAI-compatible response
        """
        if is_streaming:
            return self.stream_response(response, requested_model)
        else:
            try:
                parsed_response = self.parse_provider_response(response)
                return self.format_openai_response(parsed_response, requested_model)
            except Exception as e:
                logger.error(f"Error handling response: {str(e)}")
                return self.format_openai_error(f"Response handling error: {str(e)}", 500)
