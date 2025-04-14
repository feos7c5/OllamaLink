import logging
import json
import asyncio
import httpx
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class OllamaRequestHandler:
    def __init__(self, ollama_endpoint: str, response_handler=None, max_retries: int = 3, chunk_size: int = 3):
        self.ollama_endpoint = ollama_endpoint
        self.max_retries = max_retries
        self.chunk_size = chunk_size  # Reduced to 3 messages per chunk for better stability
        self.cloudflare_timeout = 95  # Cloudflare free has ~100 second limit, stay under it
        self.response_handler = response_handler
    
    async def _make_ollama_request(self, 
                                  request_data: Dict[str, Any], 
                                  timeout_seconds: int = 90) -> httpx.Response:
        """Make a request to Ollama API with retry logic."""
        url = f"{self.ollama_endpoint}/api/chat"
        retry_count = 0
        model = request_data.get("model", "unknown")
        
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    logger.info(f"Requesting Ollama model: {model} (retry {retry_count+1}/{self.max_retries})")
                    response = await client.post(url, json=request_data)
                    
                    if response.status_code != 200:
                        error_msg = f"Ollama returned non-200 status: {response.status_code}"
                        
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_text = error_data["error"].lower()
                                if "model not found" in error_text or "unknown model" in error_text:
                                    return {
                                        "error": {
                                            "message": f"Model '{model}' not found in Ollama. Please check available models with 'ollama list'.",
                                            "type": "model_not_found",
                                            "code": 404,
                                            "param": "model"
                                        }
                                    }
                        except Exception:
                            error_data = response.text
                            
                        logger.error(f"{error_msg}: {error_data}")
                        if response.status_code in (400, 404, 422):
                            return {
                                "error": {
                                    "message": f"Ollama API error: {error_data}",
                                    "type": "api_error",
                                    "code": response.status_code
                                }
                            }
                        
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                        
                    # For streaming requests, return the response directly
                    if request_data.get("stream", False):
                        logger.debug("Streaming response received from Ollama")
                        return response
                    
                    # For non-streaming, parse the JSON response
                    try:
                        response_data = response.json()
                        logger.debug(f"Received response from Ollama for model {model}")
                        return response_data
                    except Exception as e:
                        logger.error(f"Failed to parse Ollama response: {str(e)}")
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                        
            except httpx.TimeoutException:
                logger.warning(f"Request to Ollama timed out after {timeout_seconds}s on try {retry_count+1}/{self.max_retries}")
                retry_count += 1

                if retry_count >= self.max_retries:
                    raise
                    
                timeout_seconds = max(timeout_seconds - 5, 30)
                await asyncio.sleep(1)
                
            except httpx.ConnectError:
                logger.error(f"Connection error to Ollama API. Check if Ollama is running.")
                return {
                    "error": {
                        "message": "Cannot connect to Ollama. Please ensure Ollama is running with 'ollama serve'.",
                        "type": "connection_error",
                        "code": 503
                    }
                }
                
            except Exception as e:
                logger.error(f"Error making request to Ollama: {str(e)}")
                retry_count += 1
                
                if retry_count >= self.max_retries:
                    raise
                    
                await asyncio.sleep(1)
        
        raise Exception(f"Failed to get response from Ollama after {self.max_retries} retries")
    
    def _trim_request_size(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce the size of a request by trimming message content."""
        trimmed_request = request_data.copy()
        messages = trimmed_request.get("messages", [])
        
        if len(messages) <= 2:
            return trimmed_request
        
        # Keep system message and first/last user messages intact
        for i, msg in enumerate(messages):
            if msg.get("role") == "system" or i == len(messages) - 1:
                continue
                
            content = msg.get("content", "")
            if len(content) > 1000:
                messages[i]["content"] = content[:400] + " ... [content trimmed for performance] ... " + content[-400:]
        
        trimmed_request["messages"] = messages
        return trimmed_request
    
    def _chunk_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split messages into smaller chunks for processing."""
        system_message = None
        user_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg
            else:
                user_messages.append(msg)
        
        if len(user_messages) <= self.chunk_size:
            return [messages]

        chunked_messages = []
        for i in range(0, len(user_messages), self.chunk_size):
            if i + self.chunk_size >= len(user_messages) - 1:
                chunk = user_messages[i:]
                if system_message:
                    chunk_with_system = [system_message] + chunk
                    chunked_messages.append(chunk_with_system)
                else:
                    chunked_messages.append(chunk)
                break
            else:
                chunk = user_messages[i:i + self.chunk_size]
                if system_message:
                    chunk_with_system = [system_message] + chunk
                    chunked_messages.append(chunk_with_system)
                else:
                    chunked_messages.append(chunk)
        
        return chunked_messages
    
    async def _process_chunked_request(self, 
                                      original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request by breaking it into smaller chunks if needed."""
        messages = original_request.get("messages", [])
        
        if len(messages) <= self.chunk_size + 1:
            logger.info("Request is small enough to process directly")
            response = await self._make_ollama_request(original_request)
            
            if isinstance(response, dict):
                if "error" in response:
                    logger.error(f"Ollama API error: {response.get('error')}")
                return response
            
            if response.status_code != 200:
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        logger.error(f"Ollama API error: {error_json['error']}")
                        return error_json
                except:
                    pass
                
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama error: {response.text}")
                
            return self.response_handler.parse_ollama_response(response)
        
        logger.info(f"Breaking complex request into chunks. Total messages: {len(messages)}")
        chunked_messages = self._chunk_messages(messages)
        logger.info(f"Request will be processed in {len(chunked_messages)} chunks")
        
        current_context = []
        combined_results = {
            "prompt_eval_count": 0,
            "eval_count": 0
        }
        
        for i, chunk in enumerate(chunked_messages):
            logger.info(f"Processing chunk {i+1}/{len(chunked_messages)} with {len(chunk)} messages")
            
            if current_context:
                chunk = chunk + current_context
            
            chunk_request = original_request.copy()
            chunk_request["messages"] = chunk
            
            is_streaming = chunk_request.pop("stream", False)
            if is_streaming and i < len(chunked_messages) - 1:
                chunk_request["stream"] = False
            
            try:
                response = await self._make_ollama_request(chunk_request)
                
                if isinstance(response, dict):
                    if "error" in response:
                        logger.error(f"Chunk {i+1} failed: {response.get('error')}")
                        raise Exception(f"Chunk processing error: {response.get('error')}")
                    
                    if i == len(chunked_messages) - 1:
                        final_response = response
                        final_response["prompt_eval_count"] = combined_results["prompt_eval_count"] + final_response.get("prompt_eval_count", 0)
                        final_response["eval_count"] = combined_results["eval_count"] + final_response.get("eval_count", 0)
                        return final_response
                    
                    resp_data = response
                else:
                    if response.status_code != 200:
                        logger.error(f"Chunk {i+1} failed: {response.status_code} - {response.text}")
                        raise Exception(f"Chunk processing error: {response.text}")
                    
                    if i == len(chunked_messages) - 1:
                        if is_streaming:
                            return response  # Return the raw response for streaming
                        
                        final_response = self.response_handler.parse_ollama_response(response)
                        final_response["prompt_eval_count"] = combined_results["prompt_eval_count"] + final_response.get("prompt_eval_count", 0)
                        final_response["eval_count"] = combined_results["eval_count"] + final_response.get("eval_count", 0)
                        return final_response
                    
                    resp_data = self.response_handler.parse_ollama_response(response)
                
                combined_results["prompt_eval_count"] += resp_data.get("prompt_eval_count", 0)
                combined_results["eval_count"] += resp_data.get("eval_count", 0)
                
                if "message" in resp_data and "content" in resp_data["message"]:
                    assistant_msg = {
                        "role": "assistant",
                        "content": resp_data["message"]["content"]
                    }
                    current_context = [assistant_msg]
                    
                if i < len(chunked_messages) - 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                if current_context and i > 0:
                    logger.info("Continuing with partial context from previous chunks")
                    continue
                else:
                    raise
                
        raise Exception("Error in chunked processing flow")
    
    async def handle_chat_request(self, 
                                 request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a chat completion request, with chunking and retry support."""
        timeout = min(request_data.get("timeout", 90), self.cloudflare_timeout)
        
        message_count = len(request_data.get("messages", []))
        is_large_request = message_count > 6 or self._calculate_request_size(request_data) > 6000
        is_stream = request_data.get("stream", False)
        
        try:
            if is_stream and not is_large_request:
                logger.info("Small streaming request, processing directly")
                response = await self._make_ollama_request(request_data, timeout)
                return response
            elif is_stream and is_large_request:
                logger.info("Large streaming request, using specialized processing")
                return await self._process_large_streaming_request(request_data)
            else:
                logger.info(f"Using chunked processing for non-streaming request")
                return await self._process_chunked_request(request_data)
                
        except httpx.TimeoutException:
            logger.error(f"Request timed out after {timeout}s")
            return {
                "error": {
                    "message": f"Request timed out after {timeout} seconds. This may be due to Cloudflare timeout limits. Try simplifying your request.",
                    "type": "timeout_error",
                    "code": 504
                }
            }
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}", exc_info=True)
            return {
                "error": {
                    "message": f"Error processing request: {str(e)}",
                    "type": "processing_error",
                    "code": 500
                }
            }
    
    def _calculate_request_size(self, request_data: Dict[str, Any]) -> int:
        """Calculate approximate size of request in bytes."""
        try:
            return len(json.dumps(request_data))
        except Exception:
            total_size = 0
            for msg in request_data.get("messages", []):
                total_size += len(msg.get("content", ""))
            return total_size
    
    async def _process_large_streaming_request(self, request_data: Dict[str, Any]) -> httpx.Response:
        """Special handling for large streaming requests to avoid Cloudflare timeouts."""
        messages = request_data.get("messages", [])
        
        system_message = None
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg
                break
        
        recent_messages = messages[-min(5, len(messages)):]
        
        if len(recent_messages) < len(messages) and system_message and system_message not in recent_messages:
            recent_messages = [system_message] + recent_messages
        
        if len(recent_messages) < len(messages):
            context_message = {
                "role": "system",
                "content": f"Note: This response is based on a reduced context of {len(recent_messages)} messages instead of the original {len(messages)} to ensure timely processing. The most recent messages were prioritized."
            }
            if system_message:
                recent_messages.insert(1, context_message)
            else:
                recent_messages.insert(0, context_message)
            
            logger.info(f"Reduced streaming request from {len(messages)} to {len(recent_messages)} messages to avoid timeout")
        
        modified_request = request_data.copy()
        modified_request["messages"] = recent_messages
        
        timeout = min(request_data.get("timeout", 90), 85)  # Stay well under the 100s Cloudflare limit
        
        logger.info(f"Sending reduced streaming request with {len(recent_messages)} messages")
        return await self._make_ollama_request(modified_request, timeout) 