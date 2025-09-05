import logging
import uuid
import time
import json
import sys
import argparse
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Constants
DEFAULT_VERIFICATION_PROMPT_TOKENS = 10
DEFAULT_VERIFICATION_COMPLETION_TOKENS = 8
CURSOR_VERIFICATION_KEYWORDS = ["test", "hello", "hi", "ping", "verify", "check", "connection"]
MAX_VERIFICATION_MESSAGE_LENGTH = 20
from .router import Router
from .util import load_config, start_localhost_run_tunnel
from .handlers import (
    OllamaRequestHandler, OllamaResponseHandler,
    OpenRouterRequestHandler, OpenRouterResponseHandler,
    LlamaCppRequestHandler, LlamaCppResponseHandler
)
import uvicorn

logger = logging.getLogger(__name__)
 
tunnel_process = None
tunnel_url = None
tunnel_port = None
        
def create_api(
    ollama_endpoint=None,
    api_key=None,
    request_callback=None
):
    """Create a new FastAPI instance with all routes configured"""
    app = FastAPI(title="OllamaLink")

    # Initialize components with proper error handling
    router = None
    ollama_request_handler = None
    ollama_response_handler = None
    openrouter_request_handler = None
    openrouter_response_handler = None
    llamacpp_request_handler = None
    llamacpp_response_handler = None
    
    try:
        router = Router(ollama_endpoint=ollama_endpoint, config_path=Path("config.json"))
        logger.info("Router initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize router: {str(e)}")
        logger.error("Server will start with limited functionality")
    
    try:
        config = load_config(Path("config.json"))
        
        ollama_endpoint_url = ollama_endpoint or config.get("ollama", {}).get("endpoint", "http://localhost:11434")
        ollama_request_handler = OllamaRequestHandler(endpoint=ollama_endpoint_url)
        ollama_response_handler = OllamaResponseHandler()
        
        openrouter_config = config.get("openrouter", {})
        openrouter_api_key = openrouter_config.get("api_key", "")
        if openrouter_api_key:
            openrouter_request_handler = OpenRouterRequestHandler(
                endpoint=openrouter_config.get("endpoint", "https://openrouter.ai/api/v1"),
                api_key=openrouter_api_key
            )
        else:
            openrouter_request_handler = None
        openrouter_response_handler = OpenRouterResponseHandler()
        
        llamacpp_config = config.get("llamacpp", {})
        llamacpp_request_handler = LlamaCppRequestHandler(
            endpoint=llamacpp_config.get("endpoint", "http://localhost:8080")
        )
        llamacpp_response_handler = LlamaCppResponseHandler()
        logger.info("All handlers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize handlers: {str(e)}")
        logger.error("Server will start with limited functionality")

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
            except (ValueError, IndexError):
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
                except (json.JSONDecodeError, ValueError):
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
            if router is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                )
            models = await router.get_available_models()
            return {"data": models, "object": "list"}
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "internal_error"}}
            )
    
    @app.get("/v1/providers/status")
    async def provider_status():
        """Get status of all providers"""
        try:
            if router is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                )
            status = router.get_provider_status()
            return status
        except Exception as e:
            logger.error(f"Error getting provider status: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "internal_error"}}
            )
    
    @app.get("/api/providers/status")
    async def gui_provider_status():
        """Get provider status formatted for GUI"""
        try:
            if router is None:
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Router not initialized", "code": "service_unavailable"}}
                )
            
            status = router.get_provider_status()
            
            # Format for GUI consumption
            gui_status = {
                "providers": {
                    "ollama": {
                        "name": "Ollama",
                        "enabled": status["ollama"]["enabled"],
                        "healthy": status["ollama"]["healthy"],
                        "models": status["ollama"]["models"],
                        "endpoint": status["ollama"]["endpoint"],
                        "status": "connected" if status["ollama"]["healthy"] else "disconnected",
                        "error": None if status["ollama"]["healthy"] else "Connection failed"
                    },
                    "openrouter": {
                        "name": "OpenRouter",
                        "enabled": status["openrouter"]["enabled"],
                        "healthy": status["openrouter"]["healthy"],
                        "models": status["openrouter"]["models"],
                        "endpoint": status["openrouter"]["endpoint"],
                        "status": "connected" if status["openrouter"]["healthy"] else "disconnected",
                        "error": None if status["openrouter"]["healthy"] else "API key or connection issue"
                    },
                    "llamacpp": {
                        "name": "Llama.cpp",
                        "enabled": status["llamacpp"]["enabled"],
                        "healthy": status["llamacpp"]["healthy"],
                        "models": status["llamacpp"]["models"],
                        "endpoint": status["llamacpp"]["endpoint"],
                        "status": "connected" if status["llamacpp"]["healthy"] else "disconnected",
                        "error": None if status["llamacpp"]["healthy"] else "Server not available"
                    }
                },
                "routing": status["routing"],
                "timestamp": int(time.time())
            }
            
            return gui_status
            
        except Exception as e:
            logger.error(f"Error getting GUI provider status: {str(e)}")
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
            
            # Detect Cursor verification requests
            model = body.get("model")
            messages = body.get("messages", [])
            is_cursor_verification = False
            
            if is_cursor and model in ["gpt-4o", "gpt-4", "gpt-3.5-turbo"] and len(messages) == 1:
                # Check if it's a short test message (typical verification)
                user_message = messages[0].get("content", "").lower().strip()
                if any(keyword in user_message for keyword in CURSOR_VERIFICATION_KEYWORDS) or len(user_message) < MAX_VERIFICATION_MESSAGE_LENGTH:
                    is_cursor_verification = True
                    logger.info(f"Detected Cursor verification request for {model}: '{user_message}'")
            
            # Don't force streaming for verification requests
            if is_cursor and not is_cursor_verification and not body.get("stream", False):
                logger.info("Forcing streaming mode for Cursor client")
                body["stream"] = True
                
            stream = body.get("stream", False)
            temperature = body.get("temperature", 0.7)
            provider = body.get("provider", None) 
            
            max_tokens = body.get("max_tokens", None)
            if max_tokens is None:
                max_tokens = body.get("max_new_tokens", None)
                if max_tokens is None:
                    max_tokens = body.get("maxOutputTokens", None)
                    
            if max_tokens is not None:
                max_tokens = int(max_tokens)
            
            # Handle Cursor verification requests with direct response
            if is_cursor_verification:
                logger.info("Responding to Cursor verification with direct response")
                verification_response = {
                    "id": f"chatcmpl-{str(uuid.uuid4())[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Connection verified. OllamaLink is ready!"
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": DEFAULT_VERIFICATION_PROMPT_TOKENS,
                        "completion_tokens": DEFAULT_VERIFICATION_COMPLETION_TOKENS,
                        "total_tokens": DEFAULT_VERIFICATION_PROMPT_TOKENS + DEFAULT_VERIFICATION_COMPLETION_TOKENS
                    },
                    "system_fingerprint": "ollamalink-server"
                }
                
                if stream:
                    # Return streaming response for verification
                    async def generate_verification_stream():
                        chunk = {
                            "id": verification_response["id"],
                            "object": "chat.completion.chunk",
                            "created": verification_response["created"],
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": "Connection verified. OllamaLink is ready!"},
                                "finish_reason": "stop"
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\\n\\n"
                        yield "data: [DONE]\\n\\n"
                    
                    return StreamingResponse(
                        generate_verification_stream(),
                        media_type="text/event-stream"
                    )
                else:
                    # Return proper OpenAI format for non-streaming
                    return JSONResponse(
                        content=verification_response,
                        status_code=200
                    )
            
            # Check if explicit provider selection is requested
            if provider:
                # Use explicit provider selection
                route_result = await router.make_request_with_provider(
                    provider=provider,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            else:
                route_result = await router.make_request(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            
            # Check if router returned an error that should be passed through
            if route_result.get("error"):
                return JSONResponse(
                    status_code=route_result["error"].get("code", 500),
                    content={"error": route_result["error"]}
                )
            
            provider = route_result.get("provider")
            display_model = route_result.get("display_model", model)
            
            if route_result.get("fallback"):
                logger.info(f"Using fallback provider: {provider}")
            
            # Use router's result directly instead of separate handlers
            if route_result.get("stream"):
                # Router returned streaming result
                return StreamingResponse(
                    route_result["stream_generator"],
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"
                    }
                )
            elif route_result.get("result"):
                # Router returned non-streaming result
                result = route_result["result"]
                if isinstance(result, dict) and "error" in result:
                    return JSONResponse(
                        status_code=result["error"].get("code", 500),
                        content={"error": result["error"]}
                    )
                else:
                    # Format as OpenAI response
                    return JSONResponse(content=result)
            
            # If we reach here, router failed to handle request properly
            logger.error(f"Router failed to handle request for model {model}, provider {provider}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": "Internal routing error", "code": "routing_error"}}
            )
                
        except Exception as e:
            logger.error(f"Error processing chat completion request: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(e), "code": "invalid_request"}}
            )
    
    @app.post("/api/tunnel/start")
    async def start_tunnel(request: Request):
        """Start a localhost.run tunnel"""
        global tunnel_process, tunnel_url, tunnel_port
        
        try:
            body = await request.json()
            port = body.get("port", 8000)
            
            if tunnel_process is not None:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "Tunnel already running", "code": "tunnel_active"}}
                )
            
            logger.info(f"Starting localhost.run tunnel for port {port}...")
            
            result = await start_localhost_run_tunnel(port)
            
            if not result:
                return JSONResponse(
                    status_code=500,
                    content={"error": {"message": "Failed to start tunnel", "code": "tunnel_start_failed"}}
                )
            
            tunnel_url, tunnel_process = result
            tunnel_port = port
            
            cursor_url = f"{tunnel_url}/v1"
            
            logger.info(f"Tunnel started successfully: {tunnel_url}")
            
            return {
                "success": True,
                "tunnel_url": tunnel_url,
                "cursor_url": cursor_url,
                "port": port,
                "status": "running"
            }
            
        except Exception as e:
            logger.error(f"Error starting tunnel: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "tunnel_error"}}
            )
    
    @app.post("/api/tunnel/stop")
    async def stop_tunnel():
        """Stop the localhost.run tunnel"""
        global tunnel_process, tunnel_url, tunnel_port
        
        try:
            if tunnel_process is None:
                return JSONResponse(
                    status_code=400,
                    content={"error": {"message": "No tunnel running", "code": "no_tunnel"}}
                )
            
            logger.info("Stopping tunnel...")
            
            if hasattr(tunnel_process, 'terminate'):
                tunnel_process.terminate()
            elif hasattr(tunnel_process, 'kill'):
                tunnel_process.kill()
            
            # Reset global state
            tunnel_process = None
            tunnel_url = None
            tunnel_port = None
            
            logger.info("Tunnel stopped successfully")
            
            return {
                "success": True,
                "status": "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error stopping tunnel: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(e), "code": "tunnel_stop_error"}}
            )
    
    @app.get("/api/tunnel/status")
    async def get_tunnel_status():
        """Get current tunnel status"""
        global tunnel_process, tunnel_url, tunnel_port
        
        is_running = tunnel_process is not None
        cursor_url = f"{tunnel_url}/v1" if tunnel_url else None
        
        return {
            "running": is_running,
            "tunnel_url": tunnel_url,
            "cursor_url": cursor_url,
            "port": tunnel_port,
            "status": "running" if is_running else "stopped"
        }

    return app


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main function to run the API server"""
    parser = argparse.ArgumentParser(description='OllamaLink API Server')
    parser.add_argument('--host', default='localhost', help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'], 
                       help='Log level (default: info)')
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        config = load_config(Path("config.json"))
        app = create_api()

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

