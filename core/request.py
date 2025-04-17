import logging
import asyncio
import httpx
from typing import Dict, List, Any
from .util import estimate_message_tokens, count_tokens_in_messages

logger = logging.getLogger(__name__)

class OllamaRequestHandler:
    def __init__(self, ollama_endpoint: str, response_handler=None, max_retries: int = 3, 
                 max_tokens_per_chunk: int = 2000, chunk_overlap: int = 1):
        self.ollama_endpoint = ollama_endpoint
        self.max_retries = max_retries
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.cloudflare_timeout = 95
        self.response_handler = response_handler
        self.prefer_streaming = True
    
    async def _make_ollama_request(self, 
                                  request_data: Dict[str, Any], 
                                  timeout_seconds: int = 90) -> httpx.Response:
        """Make a request to Ollama API with retry logic."""
        url = f"{self.ollama_endpoint}/api/chat"
        retry_count = 0
        model = request_data.get("model", "unknown")
        
        is_streaming = request_data.get("stream", self.prefer_streaming)
        if is_streaming:
            request_data["stream"] = True
        
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                    logger.info(f"Requesting Ollama model: {model} (retry {retry_count+1}/{self.max_retries}) with stream={is_streaming}")
                    
                    if is_streaming:
                        response = await client.post(
                            url, 
                            json=request_data,
                            timeout=httpx.Timeout(
                                connect=10.0,
                                read=None,
                                write=10.0,
                                pool=None
                            ),
                            headers={
                                "Accept": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no"
                            }
                        )
                    else:
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
                        
                    if is_streaming:
                        logger.debug("Streaming response received from Ollama")
                        return response
                    
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
    
    def _chunk_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split messages into smaller chunks based on token count.
        Includes an overlap of messages between chunks for context preservation.
        """
        system_message = None
        regular_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg
            else:
                regular_messages.append(msg)
        
        total_tokens = sum(estimate_message_tokens(msg) for msg in messages)
        if total_tokens <= self.max_tokens_per_chunk:
            return [messages]

        chunked_messages = []
        current_chunk = []
        current_tokens = 0
        system_message_tokens = estimate_message_tokens(system_message) if system_message else 0
        
        if system_message:
            current_chunk.append(system_message)
            current_tokens = system_message_tokens
        
        for i, msg in enumerate(regular_messages):
            msg_tokens = estimate_message_tokens(msg)
            
            if current_tokens + msg_tokens > self.max_tokens_per_chunk and current_chunk:
                if not current_chunk or (len(current_chunk) == 1 and system_message in current_chunk):
                    current_chunk.append(msg)
                
                chunked_messages.append(current_chunk)
                
                current_chunk = []
                current_tokens = 0
                
                if system_message:
                    current_chunk.append(system_message)
                    current_tokens = system_message_tokens
                
                overlap_start = max(0, i - self.chunk_overlap)
                for j in range(overlap_start, i):
                    overlap_msg = regular_messages[j]
                    current_chunk.append(overlap_msg)
                    current_tokens += estimate_message_tokens(overlap_msg)
            
            current_chunk.append(msg)
            current_tokens += msg_tokens
        
        if current_chunk and (len(current_chunk) > 1 or system_message not in current_chunk):
            chunked_messages.append(current_chunk)
        
        logger.info(f"Split {len(messages)} messages into {len(chunked_messages)} chunks based on token count")
        for i, chunk in enumerate(chunked_messages):
            chunk_tokens = sum(estimate_message_tokens(msg) for msg in chunk)
            logger.info(f"Chunk {i+1}: {len(chunk)} messages, ~{chunk_tokens} tokens")
            
        return chunked_messages
    
    async def _process_chunked_request(self, 
                                      original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request by breaking it into smaller chunks if needed."""
        messages = original_request.get("messages", [])
        
        if len(messages) <= self.max_tokens_per_chunk + 1:
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
                            return response
                        
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
        token_count = self._calculate_request_size(request_data)
        is_large_request = message_count > 8 or token_count > 2500
        
        cursor_client = True
        
        if "stream" in request_data:
            is_stream = request_data.get("stream", False)
        else:
            is_stream = self.prefer_streaming if cursor_client else False
        
        request_data["stream"] = is_stream
        
        logger.info(f"Request with {message_count} messages, estimated {token_count} tokens, streaming={is_stream}")
        
        try:
            if not is_large_request:
                if is_stream:
                    logger.info("Processing streaming request directly for real-time response")
                    return await self._make_ollama_request(request_data, timeout)
                else:
                    logger.info("Processing non-streaming request directly")
                    response = await self._make_ollama_request(request_data, timeout)
                    return response
            elif is_stream:
                logger.info("Large streaming request, using specialized processing")
                return await self._process_large_streaming_request(request_data)
            else:
                logger.info(f"Using chunked processing for large non-streaming request")
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
        """
        Calculate the approximate size of a request in tokens.
        This is a fast estimation for chunking decisions, not an exact count.
        """        
        messages = request_data.get("messages", [])
        return count_tokens_in_messages(messages)
    
    async def _process_large_streaming_request(self, request_data: Dict[str, Any]) -> httpx.Response:
        """
        Special handling for large streaming requests to avoid Cloudflare timeouts.
        Using token-based chunking with overlap between message chunks.
        """
        messages = request_data.get("messages", [])
        
        system_message = None
        for msg in messages:
            if msg.get("role") == "system":
                system_message = msg
                break
        
        total_tokens = sum(estimate_message_tokens(msg) for msg in messages)
        logger.info(f"Large streaming request with {len(messages)} messages, ~{total_tokens} tokens")
        
        if total_tokens > 4000:
            logger.info(f"Request exceeds token limit ({total_tokens} tokens), reducing context")
            
            reduced_messages = []
            current_tokens = 0
            
            if system_message:
                reduced_messages.append(system_message)
                current_tokens += estimate_message_tokens(system_message)
            
            for msg in reversed(messages):
                if msg.get("role") == "system":
                    continue
                
                msg_tokens = estimate_message_tokens(msg)
                if current_tokens + msg_tokens <= self.max_tokens_per_chunk:
                    reduced_messages.insert(1 if system_message else 0, msg)
                    current_tokens += msg_tokens
                else:
                    break
            
            if len(reduced_messages) < len(messages):
                context_message = {
                    "role": "system",
                    "content": f"Note: This response is based on a reduced context of {len(reduced_messages)} " +
                              f"messages instead of the original {len(messages)} due to token limits. " +
                              "The most recent messages were prioritized."
                }
                if system_message:
                    reduced_messages.insert(1, context_message)
                else:
                    reduced_messages.insert(0, context_message)
                
                logger.info(f"Reduced streaming request from {len(messages)} to {len(reduced_messages)} messages " +
                           f"({total_tokens} to ~{current_tokens} tokens)")
            
            modified_request = request_data.copy()
            modified_request["messages"] = reduced_messages
        else:
            modified_request = request_data
        
        timeout = min(request_data.get("timeout", 90), 85)
        
        logger.info(f"Sending streaming request with {len(modified_request.get('messages', []))} messages")
        return await self._make_ollama_request(modified_request, timeout) 