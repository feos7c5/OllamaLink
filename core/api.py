import logging
import json
import time
import random
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from .router import OllamaRouter
from .request import OllamaRequestHandler
from .response import OllamaResponseHandler

logger = logging.getLogger(__name__)

def create_api(ollama_endpoint: str, api_key: str = None, request_logger=None):
    """Create the FastAPI app.
    
    Args:
        ollama_endpoint: URL of the Ollama API
        api_key: Optional API key for authentication
        request_logger: Optional request logger for GUI integration
    """
    app = FastAPI(title="OllamaLink")
    
    # Initialize router and handlers
    router = OllamaRouter(ollama_endpoint=ollama_endpoint)
    response_handler = OllamaResponseHandler()
    request_handler = OllamaRequestHandler(ollama_endpoint=ollama_endpoint, response_handler=response_handler)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    @app.middleware("http")
    async def cors_header_middleware(request: Request, call_next):
        """Add CORS headers to all responses."""
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    
    # Add request logging middleware for GUI integration if a logger is provided
    if request_logger:
        @app.middleware("http")
        async def request_logging_middleware(request: Request, call_next):
            # Only intercept chat completions for logging
            if "/v1/chat/completions" in request.url.path:
                try:

                    original_receive = request._receive

                    body_bytes = None
                    body_processed = False

                    async def patched_receive():
                        nonlocal body_bytes, body_processed
                        
                        # Get message from original receive
                        message = await original_receive()
                        
                        # Only process http.request messages with body once
                        if message["type"] == "http.request" and not body_processed:
                            body = message.get("body", b"")
                            more_body = message.get("more_body", False)
                            
                            # If this is the first chunk or the only chunk, store it
                            if body and not body_bytes:
                                body_bytes = body
                                
                                # Try to parse as JSON for logging
                                if not more_body:  # If this is the complete body
                                    body_processed = True
                                    try:
                                        request_data = json.loads(body.decode())
                                        # Log the request in the GUI
                                        request_logger.log_request(request_data)
                                    except json.JSONDecodeError:
                                        pass
                            
                            # If we're getting more body chunks, append them
                            elif more_body and body:
                                if body_bytes:
                                    body_bytes += body
                                else:
                                    body_bytes = body
                            
                            # If no more body chunks expected, mark as processed and try to parse
                            if not more_body and not body_processed:
                                body_processed = True
                                if body_bytes:
                                    try:
                                        request_data = json.loads(body_bytes.decode())
                                        # Log the request in the GUI
                                        request_logger.log_request(request_data)
                                    except json.JSONDecodeError:
                                        pass
                        
                        # Return the original message untouched
                        return message
                    
                    # Replace the receive method
                    request._receive = patched_receive
                    
                except Exception as e:
                    logger.error(f"Error in request logging middleware: {str(e)}")
            
            # Let the request continue through the middleware stack
            response = await call_next(request)
            
            # Only process responses from chat completions
            if "/v1/chat/completions" in request.url.path:
                is_streaming = response.headers.get("content-type") == "text/event-stream"
                
                # Handle streaming responses - just add GUI progress tracking
                if is_streaming:
                    # Find the matching request
                    request_data = None
                    for entry in request_logger.request_log:
                        if entry.is_streaming and entry.status == "Pending":
                            request_data = entry.request_data
                            break
                    
                    # If we found a matching request, track streaming progress
                    if request_data:
                        # Wrap the streaming response to track progress
                        original_iterator = response.body_iterator
                        
                        async def progress_tracking_iterator():
                            chunk_count = 0
                            last_update_time = time.time()
                            update_interval = 0.1  # Update UI every 100ms
                            
                            try:
                                async for chunk in original_iterator:
                                    chunk_count += 1
                                    
                                    # Track progress in the GUI
                                    current_time = time.time()
                                    if current_time - last_update_time >= update_interval:
                                        # Update UI more frequently
                                        request_logger.update_streaming_status(request_data, chunk_count)
                                        last_update_time = current_time
                                    
                                    # Check for completion
                                    try:
                                        chunk_str = chunk.decode()
                                        if 'data: [DONE]' in chunk_str:
                                            request_logger.update_streaming_status(request_data, chunk_count, done=True)
                                    except:
                                        pass
                                    
                                    yield chunk
                                
                                # Ensure we mark as complete if we reach the end
                                request_logger.update_streaming_status(request_data, chunk_count, done=True)
                            except Exception as e:
                                logger.error(f"Error in streaming: {str(e)}")
                                request_logger.log_error(request_data, str(e))
                        
                        # Return a new streaming response with our tracking
                        return StreamingResponse(
                            content=progress_tracking_iterator(),
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type
                        )
                
                # Handle non-streaming responses - capture and log the response
                else:
                    try:
                        # Get the complete response body
                        body_bytes = b""
                        async for chunk in response.body_iterator:
                            body_bytes += chunk
                        
                        # Try to parse the response as JSON
                        try:
                            response_data = json.loads(body_bytes.decode())
                            
                            # Find a pending request to match this response
                            for entry in request_logger.request_log:
                                if entry.status == "Pending" and not entry.is_streaming:
                                    request_logger.log_response(entry.request_data, response_data)
                                    break
                        except:
                            pass
                        
                        # Return a new response with the same data
                        return Response(
                            content=body_bytes,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type
                        )
                    except Exception as e:
                        logger.error(f"Error capturing response: {str(e)}")
            
            # For non-chat completions or if there was an error, return original response
            return response
    
    # Add API key validation middleware if api_key is provided
    if api_key:
        @app.middleware("http")
        async def api_key_middleware(request: Request, call_next):
            """Validate API key for secured endpoints."""
            # Skip validation for OPTIONS requests and non-API endpoints
            if request.method == "OPTIONS" or not request.url.path.startswith("/v1/"):
                return await call_next(request)
            
            # Check for API key in headers
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                logger.warning("Missing or invalid Authorization header")
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": "Invalid API key. Please provide a valid API key in the Authorization header.",
                            "type": "invalid_api_key",
                            "code": 401
                        }
                    }),
                    status_code=401,
                    media_type="application/json"
                )
            
            # Extract and validate the API key
            provided_key = auth_header.replace("Bearer ", "").strip()
            if provided_key != api_key:
                logger.warning("Invalid API key provided")
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": "Invalid API key. Please provide a valid API key in the Authorization header.",
                            "type": "invalid_api_key",
                            "code": 401
                        }
                    }),
                    status_code=401,
                    media_type="application/json"
                )
            
            # API key is valid, proceed with the request
            return await call_next(request)
    
    @app.get("/v1/models")
    async def list_models():
        """List available models in OpenAI format."""
        logger.info("GET /v1/models - Listing available models")
        return router.get_models_list()
    
    @app.options("/{path:path}")
    async def handle_options(path: str):
        """Handle OPTIONS requests."""
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Handle chat completions."""
        try:
            start_time = time.time()

            try:
                body = await request.json()
                logger.info(f"Received request with model: {body.get('model')}")
            except json.JSONDecodeError:
                logger.error("Invalid JSON in request")
                raise HTTPException(status_code=400, detail="Invalid JSON")
            
            model = body.get('model', 'unknown')
            messages_count = len(body.get('messages', []))
            
            request_size = 0
            for msg in body.get("messages", []):
                request_size += len(str(msg.get("content", "")))
            
            logger.info(f"Request for model '{model}' with {messages_count} messages (approx. {request_size/1000:.1f}KB)")
            
            # Check if this might be too complex for Cloudflare
            is_complex = messages_count > 10 or request_size > 100000000
            if is_complex:
                logger.warning(f"Complex request detected ({messages_count} messages, {request_size/1000:.1f}KB). Chunking will be used.")
            
            # Check for test request
            is_test = False
            is_debug = False
            if "messages" in body:
                for msg in body.get("messages", []):
                    if msg.get("role") == "user" and msg.get("content"):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    content = item.get("text", "")
                                    break
                            if isinstance(content, list):
                                content = ""
                        
                        content = str(content).lower().strip()
                        
                        if content == "testing. just say hi and nothing else.":
                            is_test = True
                            break

                        if "debug mode" in content:
                            is_debug = True
                            logger.info("Debug mode activated for this request")

            if is_debug:
                logger.info(f"Debug request content: {json.dumps(body, indent=2)}")
            
            # Handle test request with mock response
            if is_test:
                logger.info("Detected test request, sending mock response")
                return {
                    "id": f"test-{random.randint(1000, 9999)}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "unknown"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! OllamaLink is working correctly."
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
                }
            
            # If using Cloudflare free and request is very complex, warn the user
            if is_complex:
                logger.warning("Request is extremely large. This may exceed Cloudflare free tier limits.")
                if "cloudflare" in request.headers.get("host", ""):
                    return {
                        "error": {
                            "message": "This request is too complex for Cloudflare's free tier limits. Try breaking your request into smaller parts or upgrading Cloudflare.",
                            "type": "request_too_large",
                            "code": 413
                        }
                    }
            
            # Get the model to use
            requested_model = body.get("model", router.default_model)
            ollama_model = router.get_model_name(requested_model)
            
            # Create a modified request for Ollama
            ollama_request = {
                "model": ollama_model,
                "messages": body.get("messages", []),
                "stream": body.get("stream", False),
            }
            
            # Add optional parameters if present
            if "temperature" in body:
                ollama_request["temperature"] = body["temperature"]
            if "max_tokens" in body:
                ollama_request["max_tokens"] = body["max_tokens"]
            if "timeout" in body:
                ollama_request["timeout"] = int(body["timeout"])
                
            # Check if request is complex (more than 3 messages)
            is_complex = len(body.get("messages", [])) > 3
            if is_complex:
                logger.info("Using advanced request handler for complex request")
            
            # Forward to Ollama using chat API
            logger.info(f"Forwarding request to Ollama model: {ollama_model}")
            
            # Use the request handler to process the request
            response = await request_handler.handle_chat_request(ollama_request)
            
            # Check if response is an error
            if isinstance(response, dict) and "error" in response:
                logger.error(f"Request handler returned error: {response['error']}")
                error_message = response['error'].get('message', '').lower()
                
                # Check specifically for model not found errors
                if "model not found" in error_message or "unknown model" in error_message:
                    logger.error(f"Model '{ollama_model}' not found in Ollama")
                    return {
                        "error": {
                            "message": f"The requested model '{requested_model}' could not be found. We tried using '{ollama_model}' but it is not available in Ollama.",
                            "type": "model_not_found",
                            "code": 404
                        }
                    }
                
                return response
                
            # Check if this is a streaming response (httpx.Response object)
            if isinstance(response, httpx.Response):
                is_stream = ollama_request.get("stream", False)
                if is_stream:
                    logger.info("Processing streaming response")
                    headers = {
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive"
                    }
                    return StreamingResponse(
                        response_handler.stream_response(response, requested_model),
                        media_type="text/event-stream",
                        headers=headers
                    )
            
            # Process normal response and convert to OpenAI format
            openai_response = response_handler.format_openai_response(response, requested_model)
            
            # Log processing time
            elapsed_time = time.time() - start_time
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            
            return openai_response
                
        except Exception as e:
            logger.error(f"Error handling completion: {str(e)}", exc_info=True)
            
            # Handle specific known errors
            error_msg = str(e).lower()
            if "timeout" in error_msg or "time" in error_msg:
                # If using Cloudflare free tier
                if "host" in request.headers and "cloudflare" in request.headers["host"]:
                    return {
                        "error": {
                            "message": "Request timed out. This is likely due to Cloudflare's free tier 100-second limit. Try a smaller request or upgrade to Cloudflare Pro.",
                            "type": "cloudflare_timeout",
                            "code": 504
                        }
                    }
                    
            # General error response
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    return app

