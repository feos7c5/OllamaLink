import logging
import json
import time
import uuid
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .router import OllamaRouter
from .request import OllamaRequestHandler
from .response import OllamaResponseHandler
from .util import load_config

logger = logging.getLogger(__name__)

def create_api(
    ollama_endpoint=None,
    api_key=None,
    request_callback=None
):
    """Create a new FastAPI instance with all routes configured"""
    app = FastAPI(title="OllamaLink")

    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        config = {
            "ollama": {
                "endpoint": ollama_endpoint or "http://localhost:11434",
                "model_mappings": {"default": "llama3"}
            }
        }

    if ollama_endpoint:
        config["ollama"]["endpoint"] = ollama_endpoint

    if api_key:
        if "openai" not in config:
            config["openai"] = {}
        config["openai"]["api_key"] = api_key

    ollama_endpoint = config["ollama"]["endpoint"]

    router = OllamaRouter(ollama_endpoint=ollama_endpoint)
    
    if "thinking_mode" in config["ollama"]:
        router.thinking_mode = config["ollama"]["thinking_mode"]
    if "skip_integrity_check" in config["ollama"]:
        router.skip_integrity_check = config["ollama"]["skip_integrity_check"]

    response_handler = OllamaResponseHandler()
    
    if not hasattr(response_handler, 'format_streaming_chunk'):
        def format_streaming_chunk(chunk, model, message_id):
            """Format a streaming chunk as an OpenAI-compatible response."""
            created_time = int(time.time())
            
            if isinstance(chunk, dict) and "error" in chunk:
                return {
                    "error": {
                        "message": chunk["error"].get("message", "Unknown error"),
                        "code": chunk["error"].get("code", 500)
                    }
                }
            
            content = None
            if isinstance(chunk, dict) and "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
            
            formatted_chunk = {
                "id": message_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {}
                }]
            }
            
            if content:
                formatted_chunk["choices"][0]["delta"]["content"] = content
            
            if isinstance(chunk, dict) and chunk.get("done", False):
                formatted_chunk["choices"][0]["finish_reason"] = "stop"
            else:
                formatted_chunk["choices"][0]["finish_reason"] = None
                
            return formatted_chunk
        
        response_handler.format_streaming_chunk = format_streaming_chunk

    max_streaming_tokens = config["ollama"].get("max_streaming_tokens", 32000)
    max_tokens_per_chunk = config["ollama"].get("max_tokens_per_chunk", 8000)

    request_handler = OllamaRequestHandler(
        ollama_endpoint=ollama_endpoint,
        response_handler=response_handler,
        max_tokens_per_chunk=max_tokens_per_chunk,
        max_streaming_tokens=max_streaming_tokens
    )

    if not hasattr(request_handler, 'prepare_ollama_request'):
        def prepare_ollama_request(model, messages, temperature=0.7, max_tokens=None):
            """Prepare a request for Ollama"""
            ollama_model = router.get_model_name(model)
            
            processed_messages = router.process_messages(messages)
            
            sanitized_messages = []
            for msg in processed_messages:
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
            
            request_data = {
                "model": ollama_model,
                "messages": sanitized_messages,
                "stream": True 
            }
            
            if temperature is not None:
                request_data["temperature"] = temperature
            if max_tokens is not None:
                request_data["max_tokens"] = max_tokens
                
            return request_data
        
        request_handler.prepare_ollama_request = prepare_ollama_request
    
    if not hasattr(request_handler, 'make_ollama_request'):
        if hasattr(request_handler, '_make_ollama_request'):
            request_handler.make_ollama_request = request_handler._make_ollama_request
        else:
            async def make_ollama_request(request_data):
                """Make a non-streaming request to Ollama"""
                url = f"{ollama_endpoint}/api/chat"
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=request_data, timeout=90)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return {"error": {"message": f"Error: {response.status_code}", "code": response.status_code}}
            
            request_handler.make_ollama_request = make_ollama_request
    
    if not hasattr(request_handler, 'stream_ollama_request'):
        if hasattr(request_handler, 'handle_chat_request'):
            async def stream_ollama_request(model, messages, temperature=0.7, max_tokens=None):
                """Stream a request to Ollama"""
                request_data = request_handler.prepare_ollama_request(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                request_data["stream"] = True
                
                response = await request_handler._make_ollama_request(request_data)
                if isinstance(response, httpx.Response):
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
                else:
                    yield response
            
            request_handler.stream_ollama_request = stream_ollama_request


    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if api_key:
        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):
            if request.url.path == "/":
                return await call_next(request)
                
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Missing API key", "code": "missing_api_key"}}
                )
                
            try:
                token_type, token = auth_header.split()
                if token_type.lower() != "bearer":
                    raise ValueError("Invalid token type")
            except:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid Authorization header", "code": "invalid_auth_header"}}
                )
                
            if token != api_key:
                return JSONResponse(
                    status_code=401,
                    content={"error": {"message": "Invalid API key", "code": "invalid_api_key"}}
                )
                
            return await call_next(request)

    if request_callback:
        @app.middleware("http")
        async def request_tracking_middleware(request: Request, call_next):
            if not request.url.path.startswith("/v1/chat/completions"):
                return await call_next(request)
                
            body = await request.body()
            request_data = None
            
            if body:
                try:
                    request_data = json.loads(body)
                    request_callback({
                        "type": "request",
                        "request": request_data
                    })
                except:
                    pass
                    
            class ResponseInterceptor(StreamingResponse):
                
                def __init__(self, content, **kwargs):
                    self.is_streaming = kwargs.pop("is_streaming", False)
                    self.original_content = content
                    self.chunk_count = 0
                    self.start_time = time.time()
                    
                    if self.is_streaming:
                        async def wrapped_content():
                            if request_callback and request_data:
                                request_callback({
                                    "type": "stream_start",
                                    "request": request_data
                                })
                                
                            try:
                                async for chunk in self.original_content:
                                    self.chunk_count += 1
                                    
                                    if self.chunk_count % 10 == 0:
                                        request_callback({
                                            "type": "stream_chunk",
                                            "request": request_data,
                                            "chunk_count": self.chunk_count,
                                            "elapsed": time.time() - self.start_time
                                        })
                                        
                                    yield chunk
                                    
                                if request_callback and request_data:
                                    request_callback({
                                        "type": "stream_end",
                                        "request": request_data,
                                        "chunk_count": self.chunk_count,
                                        "elapsed": time.time() - self.start_time
                                    })
                                    
                            except Exception as e:
                                logger.error(f"Error in stream processing: {str(e)}")
                                if request_callback and request_data:
                                    request_callback({
                                        "type": "error",
                                        "request": request_data,
                                        "error": str(e)
                                    })
                                    
                        content = wrapped_content()
                        
                    super().__init__(content, **kwargs)
                    
            async def _intercept_response(response_body):
                if request_callback and request_data:
                    try:
                        if response_body.strip().startswith("data:"):
                            request_callback({
                                "type": "response",
                                "request": request_data,
                                "response": {"message": "SSE streaming response"}
                            })
                        else:
                            response_body = response_body.strip()
                            if response_body:
                                response_data = json.loads(response_body)
                                request_callback({
                                    "type": "response",
                                    "request": request_data,
                                    "response": response_data
                                })
                            else:
                                request_callback({
                                    "type": "error",
                                    "request": request_data,
                                    "error": "Empty response received"
                                })
                    except json.JSONDecodeError as json_error:
                        logger.error(f"Error parsing JSON response: {json_error} - Response body: '{response_body[:100]}...'")
                        request_callback({
                            "type": "error",
                            "request": request_data,
                            "error": f"Invalid JSON in response: {str(json_error)}"
                        })
                    except Exception as e:
                        logger.error(f"Error tracking response: {str(e)}")
                        
                return response_body
                
            modified_request = Request(
                scope=request.scope,
                receive=request._receive
            )
            
            async def modified_receive():
                data = await request._receive()
                if data["type"] == "http.request":
                    data["body"] = body
                return data
                
            modified_request._receive = modified_receive
            
            try:
                response = await call_next(modified_request)
                
                if isinstance(response, StreamingResponse):
                    is_streaming = True
                    
                    interceptor = ResponseInterceptor(
                        response.body_iterator,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type,
                        background=response.background,
                        is_streaming=is_streaming
                    )
                    
                    return interceptor
                    
                else:
                    body = b""
                    async for chunk in response.body_iterator:
                        body += chunk
                        
                    try:
                        body_str = body.decode('utf-8')
                        processed_body = await _intercept_response(body_str)
                        body = processed_body.encode('utf-8')
                    except Exception as e:
                        logger.error(f"Error processing response: {str(e)}")
                        
                    return Response(
                        content=body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type,
                        background=response.background
                    )
            except Exception as e:
                logger.error(f"Error in request processing: {str(e)}")
                if request_callback and request_data:
                    request_callback({
                        "type": "error",
                        "request": request_data,
                        "error": str(e)
                    })
                raise

    # Define routes
    @app.get("/v1")
    async def api_info():
        """API root - provides basic info"""
        return {
            "info": "OllamaLink API Bridge",
            "ollama_endpoint": router.ollama_endpoint,
            "version": "0.1.0"
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models"""
        try:
            models = router.get_available_models()
            return {"data": models, "object": "list"}
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "internal_error"}}
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Handle chat completions"""
        try:
            body = await request.json()
            
            user_agent = request.headers.get("User-Agent", "")
            is_cursor = "Cursor" in user_agent
            
            if is_cursor and not body.get("stream", False):
                logger.info("Forcing streaming mode for Cursor client")
                body["stream"] = True
                
            stream = body.get("stream", False)
            
            model = body.get("model", "gpt-3.5-turbo")
            messages = body.get("messages", [])
            temperature = body.get("temperature", 0.7)
            
            max_tokens = body.get("max_tokens", None)
            if max_tokens is None:
                max_tokens = body.get("max_new_tokens", None)
                if max_tokens is None:
                    max_tokens = body.get("maxOutputTokens", None)
                    
            if max_tokens is not None:
                max_tokens = int(max_tokens)
                        
            ollama_request = request_handler.prepare_ollama_request(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if stream:
                async def generate_streaming_response():
                    try:
                        request_id = str(uuid.uuid4())
                        
                        async for chunk in request_handler.stream_ollama_request(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        ):
                            formatted_chunk = response_handler.format_streaming_chunk(chunk, model, request_id)
                            yield f"data: {json.dumps(formatted_chunk)}\n\n"
                            
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        logger.error(f"Error in streaming: {str(e)}")
                        error_json = json.dumps({
                            "error": {
                                "message": str(e),
                                "code": "stream_error"
                            }
                        })
                        yield f"data: {error_json}\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate_streaming_response(),
                    media_type="text/event-stream"
                )
                
            else:
                try:
                    ollama_response = await request_handler.make_ollama_request(ollama_request)
                
                    formatted_response = response_handler.format_openai_response(ollama_response, model)
                    
                    return formatted_response
                    
                except Exception as e:
                    logger.error(f"Error in chat completion: {str(e)}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": {"message": str(e), "code": "completion_error"}}
                    )
                
        except Exception as e:
            logger.error(f"Error processing chat completion request: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(e), "code": "invalid_request"}}
            )

    return app

