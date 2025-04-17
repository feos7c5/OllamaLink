import logging
import json
import time
import random
from typing import Dict, Any, AsyncGenerator
import httpx
import asyncio

logger = logging.getLogger(__name__)

class OllamaResponseHandler:
    def __init__(self):
        self.cloudflare_timeout = 95
    
    def parse_ollama_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse the Ollama response, handling multi-line JSON."""
        try:
            return response.json()
        except json.JSONDecodeError:
            try:
                content = response.text.strip()
                if "\n" in content:
                    first_json = content.split("\n")[0].strip()
                    logger.info("Parsing multi-line JSON response from Ollama (using first object)")
                    return json.loads(first_json)
                else:
                    logger.warning(f"Problem parsing Ollama response, length: {len(content)}, first 100 chars: {content[:100]}...")
                    return json.loads(content)
            except Exception as e:
                logger.error(f"Failed to parse Ollama response: {str(e)}")
                return {
                    "message": {
                        "role": "assistant", 
                        "content": "I couldn't process your request due to a technical issue. Please try again with a simpler question."
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
        logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")
        
        openai_response = {
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
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": (
                    response.get("prompt_eval_count", 0) + 
                    response.get("eval_count", 0)
                )
            },
            "system_fingerprint": "ollamalink-server"
        }
        
        return openai_response
    
    async def stream_response(self, 
                           response: httpx.Response, 
                           requested_model: str) -> AsyncGenerator[str, None]:
        """Process a streaming response from Ollama into OpenAI format for real-time delivery."""
        try:
            chunk_index = 0
            message_id = f"chatcmpl-{random.randint(10000, 99999)}"
            created_time = int(time.time())
            last_yield_time = time.time()
            buffer = ""
        
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
            await asyncio.sleep(0.01)
            
            keepalive_interval = 5
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                current_time = time.time()
                
                if current_time - last_yield_time > keepalive_interval:
                    yield f": keepalive {int(current_time)}\n\n"
                    last_yield_time = current_time
                
                try:
                    chunk = json.loads(line)
                    
                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        if content:
                            event_data = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": requested_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": content
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            chunk_index += 1
                            yield f"data: {json.dumps(event_data)}\n\n"
                            last_yield_time = current_time
                            await asyncio.sleep(0.001)
                    
                    if chunk.get("done", False):
                        finish_event = {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": requested_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(finish_event)}\n\n"
                        await asyncio.sleep(0.01)
                
                except json.JSONDecodeError:
                    sublines = [l for l in line.split("\n") if l.strip()]
                    for subline in sublines:
                        try:
                            subchunk = json.loads(subline)
                            if "message" in subchunk and "content" in subchunk["message"]:
                                content = subchunk["message"]["content"]
                                if content:
                                    event_data = {
                                        "id": message_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": requested_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {
                                                    "content": content
                                                },
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    chunk_index += 1
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    last_yield_time = current_time
                                    await asyncio.sleep(0.001)
                        except json.JSONDecodeError:
                            buffer += subline
                            try:
                                subchunk = json.loads(buffer)
                                buffer = ""
                                if "message" in subchunk and "content" in subchunk["message"]:
                                    content = subchunk["message"]["content"]
                                    if content:
                                        event_data = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": requested_model,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {
                                                        "content": content
                                                    },
                                                    "finish_reason": None
                                                }
                                            ]
                                        }
                                        chunk_index += 1
                                        yield f"data: {json.dumps(event_data)}\n\n"
                                        last_yield_time = current_time
                                        await asyncio.sleep(0.001)
                            except json.JSONDecodeError:
                                if len(buffer) > 50:
                                    event_data = {
                                        "id": message_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": requested_model,
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {
                                                    "content": buffer
                                                },
                                                "finish_reason": None
                                            }
                                        ]
                                    }
                                    chunk_index += 1
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    last_yield_time = current_time
                                    buffer = ""
                                    await asyncio.sleep(0.001)
                            except Exception as e:
                                logger.error(f"Error processing buffered content: {str(e)}")
                                buffer = ""
                        except Exception as e:
                            logger.error(f"Error processing line segment: {str(e)}")
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
            
            if buffer:
                event_data = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": requested_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": buffer
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(event_data)}\n\n"
            
            logger.info(f"Stream completed with {chunk_index} chunks")
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            error_event = {
                "error": {
                    "message": f"Stream error: {str(e)}",
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n" 