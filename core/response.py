import logging
import json
import time
import random
from typing import Dict, List, Any, AsyncGenerator
import httpx

logger = logging.getLogger(__name__)

class OllamaResponseHandler:
    def __init__(self):
        self.cloudflare_timeout = 95  # Cloudflare free has ~100 second limit, stay under it
    
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
        
        # Log token usage
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")
        
        # Create OpenAI-compatible response
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
        """Process a streaming response from Ollama into OpenAI format."""
        try:
            chunk_index = 0
            message_id = f"chatcmpl-{random.randint(10000, 99999)}"
            created_time = int(time.time())
            last_yield_time = time.time()
        
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
            last_yield_time = time.time()
            buffer = ""
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                current_time = time.time()
                if current_time - last_yield_time > 45: 
                    logger.info("Sending keepalive comment to maintain connection")
                    yield f": keepalive {int(current_time)}\n\n"
                    last_yield_time = current_time
                
                # Handle potential multi-line JSON in a single line
                if "\n" in line:
                    lines = line.split("\n")
                    logger.debug(f"Splitting multi-line content into {len(lines)} parts")
                    for subline in lines:
                        if not subline.strip():
                            continue
                        try:
                            chunk = json.loads(subline)
                            
                            if "message" in chunk and "content" in chunk["message"]:
                                content_piece = chunk["message"]["content"]
                                
                                event_data = {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": requested_model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {
                                                "content": content_piece
                                            },
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                
                                chunk_index += 1
                                yield f"data: {json.dumps(event_data)}\n\n"
                                last_yield_time = time.time()
                        except:
                            logger.debug(f"Failed to process stream subline: {subline[:50]}...")
                    continue
                
                # Process single line
                try:
                    chunk = json.loads(line)
                    
                    if "message" in chunk and "content" in chunk["message"]:
                        content_piece = chunk["message"]["content"]
                        
                        event_data = {
                            "id": message_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": requested_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": content_piece
                                    },
                                    "finish_reason": None
                                }
                            ]
                        }
                        
                        chunk_index += 1
                        try:
                            yield f"data: {json.dumps(event_data)}\n\n"
                            last_yield_time = time.time()
                        except Exception as e:
                            logger.error(f"Error yielding content chunk: {str(e)}")
                    
                    # Check for done flag
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
                        try:
                            yield f"data: {json.dumps(finish_event)}\n\n"
                            last_yield_time = time.time()
                        except Exception as e:
                            logger.error(f"Error yielding finish event: {str(e)}")
                
                except json.JSONDecodeError as e:
                    buffer += line
                    if len(buffer) > 10000:  # Prevent buffer from growing too large
                        logger.warning(f"Stream buffer too large ({len(buffer)} chars), clearing")
                        buffer = ""
 
                    try:
                        chunk = json.loads(buffer)
                        buffer = ""
                        
                        if "message" in chunk and "content" in chunk["message"]:
                            content_piece = chunk["message"]["content"]
                            event_data = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": requested_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": content_piece
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            
                            chunk_index += 1
                            yield f"data: {json.dumps(event_data)}\n\n"
                            last_yield_time = time.time()
                    except:
                        logger.debug(f"Couldn't parse JSON, buffering: {line[:50]}...")
                        continue
                    
                except Exception as e:
                    logger.error(f"Error processing stream chunk: {str(e)}")
                    continue
            
            logger.info(f"Stream completed with {chunk_index} chunks")
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream processing: {str(e)}", exc_info=True)
            error_event = {
                "error": {
                    "message": f"Stream processing error: {str(e)}",
                    "type": "stream_error"
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n" 