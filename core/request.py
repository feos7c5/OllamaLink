import logging
import asyncio
import httpx
import re
from typing import Dict, List, Any
from .util import estimate_message_tokens, count_tokens_in_messages

logger = logging.getLogger(__name__)

class OllamaRequestHandler:
    def __init__(self, ollama_endpoint: str, response_handler=None, max_retries: int = 3, 
                 max_tokens_per_chunk: int = 8000, chunk_overlap: int = 1,
                 max_streaming_tokens: int = 32000):
        self.ollama_endpoint = ollama_endpoint
        self.max_retries = max_retries
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.chunk_overlap = chunk_overlap
        self.response_handler = response_handler
        self.prefer_streaming = True
        self.max_streaming_tokens = max_streaming_tokens
    
    async def _make_ollama_request(self, 
                                  request_data: Dict[str, Any], 
                                  timeout_seconds: int = 90) -> httpx.Response:
        url = f"{self.ollama_endpoint}/api/chat"
        retry_count = 0
        model = request_data.get("model", "unknown")
        
        if "messages" in request_data:
            sanitized_messages = []
            for msg in request_data["messages"]:
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
            request_data["messages"] = sanitized_messages
        
        is_streaming = request_data.get("stream", self.prefer_streaming)
        if is_streaming:
            request_data["stream"] = True
        
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                health_response = await client.get(f"{self.ollama_endpoint}/api/version")
                if health_response.status_code != 200:
                    return {"error": {"message": "Cannot connect to Ollama API", "code": 503}}
        except Exception:
            return {"error": {"message": "Cannot connect to Ollama API", "code": 503}}
        
        while retry_count < self.max_retries:
            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Requesting model: {model} (retry {retry_count+1}/{self.max_retries}) streaming={is_streaming}")
                    
                    if is_streaming:
                        timeout = httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=5.0)
                        headers = {
                            "Accept": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no"
                        }
                        response = await client.post(url, json=request_data, timeout=timeout, headers=headers)
                    else:
                        timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=5.0)
                        response = await client.post(url, json=request_data, timeout=timeout)

                    if response.status_code != 200:
                        error_msg = f"Ollama returned status: {response.status_code}"
                        
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_text = error_data["error"].lower()
                                
                                if "model not found" in error_text:
                                    return {"error": {"message": f"Model '{model}' not found", "code": 404}}
                                
                                if "unable to load model" in error_text:
                                    return {"error": {"message": f"Model '{model}' failed to load", "code": 500}}
                        except Exception:
                            error_data = response.text
                            
                        logger.error(f"{error_msg}: {error_data}")
                        
                        if 400 <= response.status_code < 500:
                            return {"error": {"message": f"API error: {error_data}", "code": response.status_code}}
                        
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                        
                    if is_streaming:
                        return response
                    
                    try:
                        response_data = response.json()
                        return response_data
                    except Exception as e:
                        logger.error(f"Failed to parse response: {str(e)}")
                        retry_count += 1
                        await asyncio.sleep(1)
                        continue
                        
            except httpx.TimeoutException:
                logger.warning(f"Request timed out after {timeout_seconds}s")
                retry_count += 1

                if retry_count >= self.max_retries:
                    return {"error": {"message": "Request timed out", "code": 504}}
                    
                await asyncio.sleep(1)
                
            except httpx.ConnectError:
                return {"error": {"message": "Cannot connect to Ollama", "code": 503}}
                
            except Exception as e:
                logger.error(f"Error making request: {str(e)}")
                retry_count += 1
                
                if retry_count >= self.max_retries:
                    return {"error": {"message": f"Request failed: {str(e)}", "code": 500}}
                    
                await asyncio.sleep(1)
        
        return {"error": {"message": "Failed after multiple attempts", "code": 500}}
    
    def _chunk_messages(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
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
        
        if current_chunk:
            chunked_messages.append(current_chunk)
        
        logger.info(f"Split {len(messages)} messages into {len(chunked_messages)} chunks")            
        return chunked_messages
    
    async def _process_chunked_request(self, original_request: Dict[str, Any]) -> Dict[str, Any]:
        messages = original_request.get("messages", [])
        
        if len(messages) <= 5 or sum(estimate_message_tokens(msg) for msg in messages) <= self.max_tokens_per_chunk:
            logger.info("Processing request directly")
            return await self._make_ollama_request(original_request)
        
        logger.info(f"Breaking request into chunks ({len(messages)} messages)")
        chunked_messages = self._chunk_messages(messages)
        
        current_context = []
        combined_results = {"prompt_eval_count": 0, "eval_count": 0}
        
        for i, chunk in enumerate(chunked_messages):
            logger.info(f"Processing chunk {i+1}/{len(chunked_messages)}")
            
            if current_context:
                chunk = current_context + chunk
            
            chunk_request = original_request.copy()
            chunk_request["messages"] = chunk
            chunk_request["stream"] = False
            
            try:
                response = await self._make_ollama_request(chunk_request)
                
                if isinstance(response, dict):
                    if "error" in response:
                        return response
                    
                    if i == len(chunked_messages) - 1:
                        final_response = response
                        final_response["prompt_eval_count"] = combined_results["prompt_eval_count"] + final_response.get("prompt_eval_count", 0)
                        final_response["eval_count"] = combined_results["eval_count"] + final_response.get("eval_count", 0)
                        return final_response
                    
                    combined_results["prompt_eval_count"] += response.get("prompt_eval_count", 0)
                    combined_results["eval_count"] += response.get("eval_count", 0)
                    
                    if "message" in response and "content" in response["message"]:
                        assistant_msg = {
                            "role": "assistant",
                            "content": response["message"]["content"]
                        }
                        current_context = [assistant_msg]
                    
                    await asyncio.sleep(0.5)
                    
                else:
                    return {"error": {"message": "Unexpected response format", "code": 500}}
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                return {"error": {"message": f"Chunk processing error: {str(e)}", "code": 500}}
                
        return {"error": {"message": "Processing error", "code": 500}}
    
    async def _process_large_streaming_request(self, request_data: Dict[str, Any]) -> httpx.Response:
        messages = request_data.get("messages", [])
        total_tokens = count_tokens_in_messages(messages)
        logger.info(f"Processing large streaming request ({len(messages)} messages, ~{total_tokens} tokens)")
        
        max_safe_tokens = min(self.max_streaming_tokens, 12000)
        
        if total_tokens <= max_safe_tokens:
            logger.info("Request within token limit")
            return await self._make_ollama_request(request_data, min(request_data.get("timeout", 90), 90))
            
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
        
        return await self._make_ollama_request(modified_request, timeout)
    
    async def handle_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
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
                return await self._make_ollama_request(request_data)
            elif is_stream:
                return await self._process_large_streaming_request(request_data)
            else:
                return await self._process_chunked_request(request_data)
                
        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            return {"error": {"message": "Error processing request", "code": 500}}
            
    def _calculate_request_size(self, request_data: Dict[str, Any]) -> int:
        return count_tokens_in_messages(request_data.get("messages", [])) 