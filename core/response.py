import logging
import json
import time
import random
from typing import Dict, Any, AsyncGenerator
import httpx

logger = logging.getLogger(__name__)

class OllamaResponseHandler:
    def __init__(self):
        pass
    
    def parse_ollama_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse the Ollama response, handling multi-line JSON."""
        try:
            return response.json()
        except json.JSONDecodeError:
            try:
                content = response.text.strip()
                if "\n" in content:
                    first_json = content.split("\n")[0].strip()
                    return json.loads(first_json)
                else:
                    return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to parse response: {str(e)}")
                return {
                    "message": {
                        "role": "assistant", 
                        "content": "I couldn't process your request. Please try again."
                    },
                    "prompt_eval_count": 0,
                    "eval_count": 0
                }
    
    def format_openai_response(self, response: Dict[str, Any], requested_model: str) -> Dict[str, Any]:
        """Format Ollama response as OpenAI-compatible response."""
        assistant_content = ""
        if "message" in response and "content" in response["message"]:
            assistant_content = response["message"]["content"]
        
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        
        return {
            "id": f"chatcmpl-{random.randint(10000, 99999)}",
            "object": "chat.completion",
            "created": int(time.time()),
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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "system_fingerprint": "ollamalink-server"
        }
    
    async def stream_response(self, 
                           response: httpx.Response, 
                           requested_model: str) -> AsyncGenerator[str, None]:
        """Process streaming response from Ollama into OpenAI format."""
        try:
            chunk_index = 0
            message_id = f"chatcmpl-{random.randint(10000, 99999)}"
            created_time = int(time.time())
            last_yield_time = time.time()
            start_time = time.time()
            
            role_event = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant"
                        },
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(role_event)}\n\n"
            
            keepalive_interval = 3
            
            max_stream_time = 300
            
            async for line in response.aiter_lines():
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                if elapsed_time > max_stream_time:
                    logger.warning(f"Stream timeout protection after {elapsed_time:.1f}s")
                    
                    timeout_message = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": "\n\n[Response truncated to prevent timeout]"
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(timeout_message)}\n\n"
                    
                    finish_event = {
                        "id": message_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": requested_model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length"
                        }]
                    }
                    yield f"data: {json.dumps(finish_event)}\n\n"
                    
                    yield "data: [DONE]\n\n"
                    return
                
                if current_time - last_yield_time > keepalive_interval:
                    yield f": keepalive {int(current_time)}\n\n"
                    last_yield_time = current_time
                
                if not line.strip():
                    continue
                
                try:
                    chunk = json.loads(line)
                    
                    if "error" in chunk:
                        logger.error(f"Stream error: {chunk['error']}")
                        error_data = {"error": {"message": "Stream error", "code": 500}}
                        yield f"data: {json.dumps(error_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if content:
                            content_data = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": requested_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {
                                        "content": content
                                    },
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(content_data)}\n\n"
                            chunk_index += 1
                            last_yield_time = current_time
                    
                    if chunk.get("done", False):
                        done_data = {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": requested_model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(done_data)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
            
            logger.info(f"Stream completed: {chunk_index} chunks in {time.time() - start_time:.2f}s")
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}")
            try:
                error_data = {"error": {"message": "Stream error", "code": 500}}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            except:
                pass 