import logging
import httpx
import json
import time
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from .clients import OllamaClient, OpenRouterClient, LlamaCppClient
from .util import load_config

logger = logging.getLogger(__name__)

class Router:
    """
    Enhanced router that supports Ollama and OpenRouter.ai providers.
    """
    
    def __init__(self, ollama_endpoint: str = None, config_path: str = "config.json"):
        self.config = load_config(config_path)
        self.ollama_client = None
        self.openrouter_client = None
        self.llamacpp_client = None
        
        # Initialize Ollama client if enabled
        ollama_config = self.config.get("ollama", {})
        if ollama_config.get("enabled", True):  # Default to enabled for backwards compatibility
            ollama_endpoint = ollama_endpoint or ollama_config.get("endpoint", "http://localhost:11434")
            self.ollama_endpoint = ollama_endpoint 
            self.ollama_client = OllamaClient(endpoint=ollama_endpoint)
            self.thinking_mode = ollama_config.get("thinking_mode", True)
        else:
            self.ollama_client = None
            self.ollama_endpoint = None
            self.thinking_mode = False
        
        # Initialize OpenRouter client if enabled
        openrouter_config = self.config.get("openrouter", {})
        if openrouter_config.get("enabled", False):
            api_key = openrouter_config.get("api_key")
            if api_key:
                self.openrouter_client = OpenRouterClient(
                    api_key=api_key,
                    endpoint=openrouter_config.get("endpoint", "https://openrouter.ai/api/v1")
                )
                logger.info("OpenRouter client initialized")
            else:
                logger.warning("OpenRouter enabled but no API key provided")
        
        # Initialize Llama.cpp client if enabled
        llamacpp_config = self.config.get("llamacpp", {})
        if llamacpp_config.get("enabled", False):
            self.llamacpp_client = LlamaCppClient(
                endpoint=llamacpp_config.get("endpoint", "http://localhost:8080")
            )
            logger.info("Llama.cpp client initialized")
        
        # Load model mappings
        self.ollama_mappings = self.config.get("ollama", {}).get("model_mappings", {})
        self.openrouter_mappings = self.config.get("openrouter", {}).get("model_mappings", {})
        self.llamacpp_mappings = self.config.get("llamacpp", {}).get("model_mappings", {})
        
        # Routing configuration
        routing_config = self.config.get("routing", {})
        self.routing_config = routing_config
        self.provider_priority = routing_config.get("provider_priority", ["ollama", "llamacpp", "openrouter"])
        self.enable_fallback = routing_config.get("enable_fallback", True)
        self.cost_optimization = routing_config.get("cost_optimization", True)
        
        # Provider health tracking
        self.provider_health = {
            "ollama": {"available": bool(self.ollama_client), "last_check": 0, "consecutive_failures": 0},
            "openrouter": {"available": bool(self.openrouter_client), "last_check": 0, "consecutive_failures": 0},
            "llamacpp": {"available": bool(self.llamacpp_client), "last_check": 0, "consecutive_failures": 0}
        }
        
        # Initialize models for all clients
        if self.ollama_client:
            try:
                models = self.ollama_client.fetch_models()
                logger.info(f"Successfully fetched {len(models)} models from Ollama")
                if self.ollama_client.connection_error:
                    logger.error(f"Ollama connection error: {self.ollama_client.connection_error}")
            except Exception as e:
                logger.error(f"Failed to fetch Ollama models during initialization: {str(e)}")
                
        if self.llamacpp_client:
            try:
                models = self.llamacpp_client.fetch_models()
                logger.info(f"Successfully fetched {len(models)} models from Llama.cpp")
            except Exception as e:
                logger.error(f"Failed to fetch Llama.cpp models during initialization: {str(e)}")
        
        ollama_models = len(self.ollama_client.available_models) if self.ollama_client else 0
        logger.info(f"Router initialized - Ollama: {ollama_models} models, OpenRouter: {bool(self.openrouter_client)}, Llama.cpp: {bool(self.llamacpp_client)}")
        logger.info(f"Provider priority: {self.provider_priority}")
        logger.info(f"OpenRouter mappings: {self.openrouter_mappings}")
        logger.info(f"Ollama mappings: {self.ollama_mappings}")
    

    
    async def _make_ollama_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a direct HTTP request to Ollama."""
        url = f"{self.ollama_endpoint}/api/chat"
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(url, json=request_data)
                
                if response.status_code == 200:
                    if request_data.get("stream", False):
                        return {"status": "streaming", "response": response}
                    else:
                        return {"status": "success", "data": response.json()}
                else:
                    return {
                        "status": "error",
                        "code": response.status_code,
                        "error": {"message": f"Ollama returned status: {response.status_code}"}
                    }
        except Exception as e:
            logger.error(f"Ollama request failed: {str(e)}")
            return {
                "status": "error",
                "error": {"message": f"Request failed: {str(e)}"}
            }
    
    async def _stream_ollama_request(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a request to Ollama."""
        url = f"{self.ollama_endpoint}/api/chat"
        
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                async with client.stream("POST", url, json=request_data) as response:
                    if response.status_code != 200:
                        yield {
                            "error": {"message": f"Ollama returned status: {response.status_code}", "code": response.status_code}
                        }
                        return
                    
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            logger.error(f"Ollama streaming failed: {str(e)}")
            yield {
                "error": {"message": f"Streaming failed: {str(e)}"}
            }
    
    async def determine_provider_and_model(self, requested_model: str) -> Tuple[str, str, str]:
        """
        Determine the best provider and model for a request.
        
        Returns:
            Tuple of (provider, actual_model, display_model)
        """
        logger.info(f"Determining provider for model: {requested_model}")
        logger.info(f"Checking mappings - Ollama: {requested_model in self.ollama_mappings}, OpenRouter: {requested_model in self.openrouter_mappings}")
        
        # Check if model is explicitly mapped to a provider
        if requested_model in self.ollama_mappings:
            ollama_model = self.ollama_mappings[requested_model]
            ollama_healthy = await self._is_provider_healthy("ollama")
            logger.info(f"Ollama mapping found: {ollama_model}, healthy: {ollama_healthy}")
            if ollama_healthy:
                return "ollama", ollama_model, requested_model
        
        if requested_model in self.openrouter_mappings:
            openrouter_model = self.openrouter_mappings[requested_model]
            openrouter_healthy = await self._is_provider_healthy("openrouter")
            logger.info(f"OpenRouter mapping found: {openrouter_model}, healthy: {openrouter_healthy}")
            if openrouter_healthy:
                return "openrouter", openrouter_model, requested_model
        
        # Use provider priority order
        for provider in self.routing_config.get("provider_priority", ["ollama", "llamacpp", "openrouter"]):
            if await self._is_provider_healthy(provider):
                if provider == "ollama":
                    actual_model = self.ollama_client.get_model_name(requested_model, self.ollama_mappings)
                    return "ollama", actual_model, requested_model
                elif provider == "llamacpp" and self.llamacpp_client:
                    actual_model = self.llamacpp_client.get_model_name(requested_model, self.llamacpp_mappings)
                    return "llamacpp", actual_model, requested_model
                elif provider == "openrouter" and self.openrouter_client:
                    # Check if model exists in OpenRouter
                    if not self.openrouter_client.available_models:
                        await self.openrouter_client.fetch_models()
                    
                    # Try exact match first
                    for model in self.openrouter_client.available_models:
                        if model["id"].lower() == requested_model.lower():
                            return "openrouter", model["id"], requested_model
                    
                    # Try partial match
                    matching_models = self.openrouter_client.search_models(requested_model)
                    if matching_models:
                        return "openrouter", matching_models[0]["id"], requested_model
        
        # Fallback to default
        if await self._is_provider_healthy("ollama"):
            default_model = self.ollama_client.get_model_name("default", self.ollama_mappings)
            return "ollama", default_model, requested_model
        elif await self._is_provider_healthy("openrouter"):
            return "openrouter", "openai/gpt-3.5-turbo", requested_model
        
        raise Exception("No healthy providers available")
    
    async def _is_provider_healthy(self, provider: str) -> bool:
        """Check if a provider is healthy and available."""
        current_time = time.time()
        health_info = self.provider_health.get(provider, {})
        
        if current_time - health_info.get("last_check", 0) < 30:
            return health_info.get("available", False)
        
        if provider == "ollama":
            if not self.ollama_client:
                # Ollama is disabled
                self.provider_health["ollama"].update({
                    "available": False,
                    "last_check": current_time,
                    "consecutive_failures": 999  # Mark as permanently disabled
                })
                return False
            try:
                is_healthy = (len(self.ollama_client.available_models) > 0 and self.ollama_client.connection_error is None)
                self.provider_health["ollama"].update({
                    "available": is_healthy,
                    "last_check": current_time,
                    "consecutive_failures": 0 if is_healthy else health_info.get("consecutive_failures", 0) + 1
                })
                return is_healthy
            except Exception:
                self.provider_health["ollama"]["available"] = False
                return False
                
        elif provider == "openrouter" and self.openrouter_client:
            # OpenRouter is healthy if client is initialized (has API key)
            logger.info(f"OpenRouter client available: True")
            self.provider_health["openrouter"].update({
                "available": True,
                "last_check": current_time,
                "consecutive_failures": 0
            })
            return True
                
        elif provider == "llamacpp" and self.llamacpp_client:
            try:
                is_healthy = (len(self.llamacpp_client.available_models) > 0 and self.llamacpp_client.connection_error is None)
                self.provider_health["llamacpp"].update({
                    "available": is_healthy,
                    "last_check": current_time,
                    "consecutive_failures": 0 if is_healthy else health_info.get("consecutive_failures", 0) + 1
                })
                return is_healthy
            except Exception:
                self.provider_health["llamacpp"]["available"] = False
                return False
        
        return False
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get combined list of available models from all providers."""
        models = []
        
        if await self._is_provider_healthy("ollama"):
            ollama_models = self.ollama_client.get_available_models()
            for model in ollama_models:
                model["provider"] = "ollama"
                models.append(model)
        
        if await self._is_provider_healthy("openrouter"):
            try:
                openrouter_response = await self.openrouter_client.fetch_models()
                openrouter_models = openrouter_response.get("models", [])
                for model in openrouter_models:
                    models.append({
                        "id": model["id"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": model.get("owned_by", "openrouter"),
                        "provider": "openrouter",
                        "context_length": model.get("context_length", 4096),
                        "description": model.get("description", "")
                    })
            except Exception as e:
                logger.error(f"Failed to fetch OpenRouter models: {str(e)}")
        
        if await self._is_provider_healthy("llamacpp"):
            llamacpp_models = self.llamacpp_client.get_available_models()
            for model in llamacpp_models:
                model["provider"] = "llamacpp"
                models.append(model)
        
        for mapped_name in self.ollama_mappings.keys():
            if mapped_name != "default" and not any(m["id"] == mapped_name for m in models):
                models.append({
                    "id": mapped_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama-mapped",
                    "provider": "ollama"
                })
        
        for mapped_name in self.openrouter_mappings.keys():
            if not any(m["id"] == mapped_name for m in models):
                models.append({
                    "id": mapped_name,
                    "object": "model", 
                    "created": int(time.time()),
                    "owned_by": "openrouter-mapped",
                    "provider": "openrouter"
                })
        
        for mapped_name in self.llamacpp_mappings.keys():
            if not any(m["id"] == mapped_name for m in models):
                models.append({
                    "id": mapped_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "llamacpp-mapped",
                    "provider": "llamacpp"
                })
        
        return models
    
    async def make_request_with_provider(self, provider: str, model: str, messages: List[Dict[str, Any]], 
                                       temperature: float = 0.7, max_tokens: Optional[int] = None,
                                       stream: bool = False) -> Dict[str, Any]:
        """
        Make a request to a specific provider explicitly chosen by the frontend.
        """
        logger.info(f"Explicit provider request: {provider} for model {model}")
        
        if provider == "ollama":
            return await self._make_ollama_request_direct(model, messages, temperature, max_tokens, stream)
        elif provider == "openrouter":
            return await self._make_openrouter_request_direct(model, messages, temperature, max_tokens, stream)
        elif provider == "llamacpp":
            return await self._make_llamacpp_request_direct(model, messages, temperature, max_tokens, stream)
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported providers: ollama, openrouter, llamacpp")
    
    async def _make_ollama_request_direct(self, model: str, messages: List[Dict[str, Any]], 
                                        temperature: float = 0.7, max_tokens: Optional[int] = None,
                                        stream: bool = False) -> Dict[str, Any]:
        """Make a direct request to Ollama."""
        if not self.ollama_client:
            raise Exception("Ollama client not available")
        
        # Get the actual model name using Ollama mappings
        actual_model = self.ollama_client.get_model_name(model, self.ollama_mappings)
        processed_messages = self.ollama_client.process_messages(messages, self.thinking_mode)
        
        if stream:
            return {
                "provider": "ollama",
                "model": actual_model,
                "display_model": model,
                "stream": True,
                "stream_generator": self.ollama_client.stream_chat_completion(
                    model=actual_model,
                    messages=processed_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            }
        else:
            result = await self.ollama_client.chat_completion(
                model=actual_model,
                messages=processed_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return {
                "provider": "ollama",
                "model": actual_model,
                "display_model": model,
                "result": result,
                "stream": False
            }
    
    async def _make_openrouter_request_direct(self, model: str, messages: List[Dict[str, Any]], 
                                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                                            stream: bool = False) -> Dict[str, Any]:
        """Make a direct request to OpenRouter."""
        if not self.openrouter_client:
            raise Exception("OpenRouter client not available")
        
        # Get the actual model name using OpenRouter mapping
        actual_model = model
        if model in self.openrouter_mappings:
            actual_model = self.openrouter_mappings[model]
        
        if stream:
            return {
                "provider": "openrouter",
                "model": actual_model,
                "display_model": model,
                "stream": True,
                "stream_generator": self.openrouter_client.stream_chat_completion(
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            }
        else:
            result = await self.openrouter_client.chat_completion(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return {
                "provider": "openrouter",
                "model": actual_model,
                "display_model": model,
                "result": result,
                "stream": False
            }
    
    async def _make_llamacpp_request_direct(self, model: str, messages: List[Dict[str, Any]], 
                                          temperature: float = 0.7, max_tokens: Optional[int] = None,
                                          stream: bool = False) -> Dict[str, Any]:
        """Make a direct request to Llama.cpp."""
        if not self.llamacpp_client:
            raise Exception("Llama.cpp client not available")
        
        # Get the actual model name using Llama.cpp mappings
        actual_model = self.llamacpp_client.get_model_name(model, self.llamacpp_mappings)
        processed_messages = self.llamacpp_client.process_messages(messages, self.thinking_mode)
        
        if stream:
            return {
                "provider": "llamacpp",
                "model": actual_model,
                "display_model": model,
                "stream": True,
                "stream_generator": self.llamacpp_client.stream_chat_completion(
                    model=actual_model,
                    messages=processed_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            }
        else:
            result = await self.llamacpp_client.chat_completion(
                model=actual_model,
                messages=processed_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            return {
                "provider": "llamacpp",
                "model": actual_model,
                "display_model": model,
                "result": result,
                "stream": False
            }
    
    async def make_request(self, model: str, messages: List[Dict[str, Any]], 
                          temperature: float = 0.7, max_tokens: Optional[int] = None,
                          stream: bool = False) -> Dict[str, Any]:
        """
        Route request to appropriate provider with fallback support.
        """
        primary_error = None
        
        try:
            provider, actual_model, display_model = await self.determine_provider_and_model(model)
            logger.info(f"Routing {model} → {provider}:{actual_model}")
            
            if provider == "ollama":
                # Use Ollama with direct HTTP handling
                processed_messages = self.ollama_client.process_messages(messages, self.thinking_mode)
                
                # Create request data in Ollama format
                request_data = {
                    "model": actual_model,
                    "messages": processed_messages,
                    "temperature": temperature,
                    "stream": stream
                }
                
                if max_tokens:
                    request_data["max_tokens"] = max_tokens
                
                if stream:
                    return {
                        "provider": provider,
                        "model": actual_model,
                        "display_model": display_model,
                        "stream": True,
                        "stream_generator": self._stream_ollama_request(request_data)
                    }
                else:
                    result = await self._make_ollama_request(request_data)
                    return {
                        "provider": provider,
                        "model": actual_model,
                        "display_model": display_model,
                        "result": result,
                        "stream": False
                    }
                
            elif provider == "openrouter":
                # Use OpenRouter client
                if stream:
                    return {
                        "provider": provider,
                        "model": actual_model,
                        "display_model": display_model,
                        "stream": True,
                        "stream_generator": self.openrouter_client.stream_chat_completion(
                            model=actual_model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                    }
                else:
                    result = await self.openrouter_client.chat_completion(
                        model=actual_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False
                    )
                    
                    # Check if result is an error response - don't fallback for user/billing errors
                    if isinstance(result, dict) and "error" in result:
                        error_code = result["error"].get("code", 500)
                        # Don't fallback for user errors (401, 402, 403, 429) - return them directly
                        if error_code in [401, 402, 403, 429]:
                            return {
                                "provider": provider,
                                "model": actual_model,
                                "display_model": display_model,
                                "result": result,
                                "stream": False,
                                "error": result["error"]  # Include error for API handler
                            }
                    
                    return {
                        "provider": provider,
                        "model": actual_model,
                        "display_model": display_model,
                        "result": result,
                        "stream": False
                    }
                    
        except Exception as e:
            primary_error = e
            logger.error(f"Primary provider failed: {str(e)}")
        
        # Fallback logic
        if self.routing_config.get("fallback_enabled", True) and primary_error:
            logger.info("Attempting fallback to alternative provider")
            
            try:
                # Try the other provider
                alternative_providers = ["ollama", "openrouter"]
                for alt_provider in alternative_providers:
                    if await self._is_provider_healthy(alt_provider):
                        if alt_provider == "ollama":
                            fallback_model = self.ollama_client.get_model_name(model, self.ollama_mappings)
                            logger.info(f"Fallback: {model} → ollama:{fallback_model}")
                            
                            processed_messages = self.ollama_client.process_messages(messages, self.thinking_mode)
                            request_data = {
                                "model": fallback_model,
                                "messages": processed_messages,
                                "temperature": temperature,
                                "stream": stream
                            }
                            
                            if max_tokens:
                                request_data["max_tokens"] = max_tokens
                            
                            if stream:
                                return {
                                    "provider": "ollama",
                                    "model": fallback_model,
                                    "display_model": model,
                                    "stream": True,
                                    "stream_generator": self._stream_ollama_request(request_data),
                                    "fallback": True
                                }
                            else:
                                result = await self._make_ollama_request(request_data)
                                return {
                                    "provider": "ollama",
                                    "model": fallback_model,
                                    "display_model": model,
                                    "result": result,
                                    "stream": False,
                                    "fallback": True
                                }
                            
                        elif alt_provider == "openrouter" and self.openrouter_client:
                            # Use a common fallback model
                            fallback_model = "openai/gpt-3.5-turbo"
                            logger.info(f"Fallback: {model} → openrouter:{fallback_model}")
                            
                            if stream:
                                return {
                                    "provider": "openrouter",
                                    "model": fallback_model,
                                    "display_model": model,
                                    "stream": True,
                                    "fallback": True,
                                    "stream_generator": self.openrouter_client.stream_chat_completion(
                                        model=fallback_model,
                                        messages=messages,
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                }
                            else:
                                result = await self.openrouter_client.chat_completion(
                                    model=fallback_model,
                                    messages=messages,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    stream=False
                                )
                                
                                return {
                                    "provider": "openrouter",
                                    "model": fallback_model,
                                    "display_model": model,
                                    "result": result,
                                    "stream": False,
                                    "fallback": True
                                }
                        break
                        
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
        
        # All providers failed
        raise Exception(f"All providers failed. Primary error: {str(primary_error)}")
    
    def get_provider_status(self) -> Dict[str, Any]:
        """Get status of all providers."""
        return {
            "ollama": {
                "enabled": True,
                "healthy": self.provider_health["ollama"]["available"],
                "models": len(self.ollama_client.available_models) if self.ollama_client else 0,
                "endpoint": self.ollama_endpoint,
                "consecutive_failures": self.provider_health["ollama"]["consecutive_failures"]
            },
            "openrouter": {
                "enabled": bool(self.openrouter_client),
                "healthy": self.provider_health["openrouter"]["available"],
                "models": len(self.openrouter_client.available_models) if self.openrouter_client else 0,
                "endpoint": self.openrouter_client.endpoint if self.openrouter_client else None,
                "consecutive_failures": self.provider_health["openrouter"]["consecutive_failures"]
            },
            "llamacpp": {
                "enabled": bool(self.llamacpp_client),
                "healthy": self.provider_health["llamacpp"]["available"],
                "models": len(self.llamacpp_client.available_models) if self.llamacpp_client else 0,
                "endpoint": self.llamacpp_client.endpoint if self.llamacpp_client else None,
                "consecutive_failures": self.provider_health["llamacpp"]["consecutive_failures"]
            },
            "routing": self.routing_config
        }
