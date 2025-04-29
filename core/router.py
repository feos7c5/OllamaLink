import logging
import requests
from typing import Dict
import time
import re
import json
import os

logger = logging.getLogger(__name__)

class OllamaRouter:
    def __init__(self, ollama_endpoint: str, config_path: str = "config.json"):
        self.ollama_endpoint = ollama_endpoint
        self.available_models = []
        self.default_model = None
        self.model_mappings = {}
        self.connection_error = None
        self.model_error = None
        self.thinking_mode = True 
        self.skip_integrity_check = False
        self._load_config(config_path)
        self._fetch_models()
        
    def _load_config(self, config_path: str) -> None:
        """Load configuration from config.json."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if "ollama" in config:
                    if "model_mappings" in config["ollama"]:
                        self.model_mappings = config["ollama"]["model_mappings"]
                        logger.info(f"Loaded model mappings from config: {self.model_mappings}")
                        
                        if "default" in self.model_mappings:
                            self.default_model = self.model_mappings["default"]
                            logger.info(f"Set default model from config: {self.default_model}")
                            
                    # Load thinking mode setting
                    if "thinking_mode" in config["ollama"]:
                        self.thinking_mode = config["ollama"]["thinking_mode"]
                        logger.info(f"Thinking mode set to: {self.thinking_mode}")
                    
                    # Load integrity check setting
                    if "skip_integrity_check" in config["ollama"]:
                        self.skip_integrity_check = config["ollama"]["skip_integrity_check"]
                        if self.skip_integrity_check:
                            logger.info("Model integrity checks are disabled")
            else:
                logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
    
    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name for consistent comparison."""
        name = re.sub(r':[^:]+$', '', name)
        return name.lower()
    
    def _fetch_models(self):
        """Get available models from Ollama."""
        self.connection_error = None
        self.model_error = None
        
        try:
            logger.info(f"Fetching models from Ollama at {self.ollama_endpoint}")
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if "models" in data and len(data["models"]) > 0:
                    self.available_models = [m["name"] for m in data["models"] if "name" in m]
                    
                    self.available_models = sorted(self.available_models)
                    
                    if not self.default_model and self.available_models:
                        if hasattr(self, 'model_mappings') and "default" in self.model_mappings:
                            default_from_config = self.model_mappings["default"]
                            
                            if default_from_config in self.available_models:
                                self.default_model = default_from_config
                            else:
                                for model in self.available_models:
                                    if (self._normalize_model_name(model) == self._normalize_model_name(default_from_config)):
                                        self.default_model = model
                                        break
                        
                        if not self.default_model:
                            self.default_model = self.available_models[0]
                    
                    logger.info(f"Default model: {self.default_model}")
                    logger.info(f"Found {len(self.available_models)} models")
                    logger.debug(f"Models: {', '.join(self.available_models)}")
                    
                    if self.default_model and not self.skip_integrity_check:
                        self._check_model_integrity(self.default_model)
                    elif self.skip_integrity_check:
                        logger.info("Skipping model integrity check as configured")
                    
                    return
                else:
                    error_msg = "No models found in Ollama"
                    logger.warning(error_msg)
                    self.connection_error = error_msg
            else:
                error_msg = f"Failed to get models from Ollama: HTTP {response.status_code}"
                logger.warning(error_msg)
                
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_text = error_data["error"].lower()
                        if "unable to load model" in error_text and "/blobs/sha256-" in error_text:
                            blob_pattern = r'(/.*?/models/blobs/sha256-[a-f0-9]+)'
                            match = re.search(blob_pattern, error_data["error"])
                            blob_path = match.group(1) if match else "a model blob"
                            
                            self.model_error = {
                                "message": f"Model has a corrupted data file. Try removing and reinstalling the affected model.",
                                "type": "model_corrupted",
                                "details": f"Corrupted blob at {blob_path}"
                            }
                            logger.error(f"Model corruption detected: {blob_path}")
                        else:
                            self.connection_error = error_msg
                except Exception:
                    self.connection_error = error_msg
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to Ollama at {self.ollama_endpoint}. Is Ollama running?"
            logger.error(error_msg)
            self.connection_error = error_msg
        except requests.exceptions.Timeout:
            error_msg = f"Connection to Ollama timed out. Check if Ollama is running properly."
            logger.error(error_msg)
            self.connection_error = error_msg
        except Exception as e:
            error_msg = f"Error connecting to Ollama: {str(e)}"
            logger.error(error_msg)
            self.connection_error = error_msg
        
        if hasattr(self, 'model_mappings') and "default" in self.model_mappings:
            self.default_model = self.model_mappings["default"]
            self.available_models = [self.default_model]
            logger.warning(f"Using configured default as fallback model: {self.default_model}")
            logger.warning("Model mappings will not work until Ollama connection is restored.")
        else:
            self.default_model = "Qwen2.5-Coder:latest"
            self.available_models = [self.default_model]
            logger.warning(f"Using hardcoded fallback model: {self.default_model}")
            logger.warning("Please start Ollama or ensure it's properly configured.")
            
    def _check_model_integrity(self, model_name):
        """
        Check if a model can be loaded by sending a minimal request.
        Used to detect corrupted model blobs early during initialization.
        """
        try:
            if ":latest" in model_name:
                logger.info(f"Skipping integrity check for {model_name} (has :latest tag)")
                return
                
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "test"}],
                "stream": False
            }
            
            test_url = f"{self.ollama_endpoint}/api/chat"
            
            response = requests.post(test_url, json=test_payload, timeout=10)
            
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_text = error_data["error"].lower()
                        if "unable to load model" in error_text and "/blobs/sha256-" in error_text:
                            blob_pattern = r'(/.*?/models/blobs/sha256-[a-f0-9]+)'
                            match = re.search(blob_pattern, error_data["error"])
                            blob_path = match.group(1) if match else "a model blob"
                            
                            self.model_error = {
                                "message": f"Model '{model_name}' has a corrupted data file. Try removing and reinstalling the model with 'ollama rm {model_name}' followed by 'ollama pull {model_name}'.",
                                "type": "model_corrupted",
                                "details": f"Corrupted blob at {blob_path}"
                            }
                            logger.error(f"Model corruption detected in {model_name}: {blob_path}")
                except Exception:
                    pass
        except requests.exceptions.Timeout:
            logger.info(f"Model integrity check timed out for {model_name} - this is normal for large models")
        except requests.exceptions.ConnectionError:
            logger.info(f"Connection error during integrity check for {model_name} - Ollama might be initializing")
        except Exception as e:
            logger.warning(f"Model integrity check failed: {str(e)}")
            
    
    def get_model_name(self, requested_model: str) -> str:
        """Map requested model to available Ollama model."""
        logger.info(f"Looking for model match: {requested_model}")
        
        if not requested_model:
            logger.warning("Empty model name requested, using default model")
            return self.default_model
            
        if not self.available_models:
            logger.warning("No models available, returning requested model as-is")
            return requested_model
            
        if hasattr(self, 'connection_error') and self.connection_error:
            logger.warning(f"Cannot map model due to connection error: {self.connection_error}")
            return self.default_model
        
        if requested_model in self.available_models:
            logger.info(f"Exact match found for {requested_model}")
            return requested_model
            
        normalized_requested = self._normalize_model_name(requested_model)
        for model in self.available_models:
            if self._normalize_model_name(model) == normalized_requested:
                logger.info(f"Normalized match found: {model}")
                return model
        
        base_model_match = re.match(r'^([a-zA-Z0-9_-]+(?:-[a-zA-Z0-9_-]+)*)(?:-\d+.*)?$', normalized_requested)
        if base_model_match:
            base_model = base_model_match.group(1)
            for model in self.available_models:
                normalized_model = self._normalize_model_name(model)
                if normalized_model.startswith(base_model):
                    logger.info(f"Base model match: {requested_model} → {model}")
                    return model
        
        if self.model_mappings:
            if requested_model in self.model_mappings and isinstance(self.model_mappings[requested_model], str):
                target_model = self.model_mappings[requested_model]
                norm_target = self._normalize_model_name(target_model)
                
                for available_model in self.available_models:
                    if self._normalize_model_name(available_model) == norm_target:
                        logger.info(f"Model mapping exact match: {requested_model} → {available_model}")
                        return available_model
                
                for available_model in self.available_models:
                    normalized_available = self._normalize_model_name(available_model)
                    if normalized_available.startswith(norm_target) or normalized_available == norm_target:
                        logger.info(f"Model mapping match: {requested_model} → {available_model}")
                        return available_model
        
        standard_prefixes = ("gpt-", "claude-", "llama-", "gemini-", "mistral-", "meta-", "qwen-", "phi-")
        if any(normalized_requested.startswith(prefix) for prefix in standard_prefixes):
            logger.info(f"Standard model name {requested_model} → using default: {self.default_model}")
            return self.default_model
        
        if normalized_requested.startswith("gpt-"):
            base_name = normalized_requested[4:]
            for model in self.available_models:
                if base_name in self._normalize_model_name(model):
                    logger.info(f"Found model containing {base_name}: {model}")
                    return model
        
        logger.warning(f"No match found for {requested_model}, using default: {self.default_model}")
        logger.warning(f"This model mapping might not work if {self.default_model} isn't loaded in Ollama")
        return self.default_model
    
    def get_available_models(self) -> list:
        """Get list of available models."""
        models_list = []
        
        for api_model, _ in self.model_mappings.items():
            if api_model != "default":
                models_list.append({
                    "id": api_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
        
        for model in self.available_models:
            if model not in [m.get("id") for m in models_list]:
                models_list.append({
                    "id": model,
                    "object": "model", 
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
        
        return models_list
    
    def get_models_list(self) -> Dict:
        """Get list of models in OpenAI-compatible format."""
        try:
            models_list = self.get_available_models()
            logger.info(f"Returning {len(models_list)} models to client")
            return {
                "object": "list",
                "data": models_list
            }
        except Exception as e:
            logger.error(f"Error getting models list: {str(e)}")
            return {
                "object": "list",
                "data": []
            }
    
    def process_messages(self, messages):
        """
        Process messages based on thinking mode setting.
        
        Thinking mode controls whether models should "think" before responding:
        - When enabled (default): Messages are passed through unchanged
        - When disabled: The /no_think prefix is automatically added to user messages
        
        The /no_think prefix is a special command recognized by Ollama models that
        instructs them to skip the "thinking" step and respond more directly.
        This typically results in faster responses but potentially less thorough analysis.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Processed list of messages
        """
        if self.thinking_mode:
            return messages
            
        processed_messages = []
        for message in messages:
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                
                if isinstance(content, str) and not content.startswith("/no_think"):
                    message = message.copy()
                    message["content"] = f"/no_think {content}"
                    logger.info(f"Added /no_think prefix to user message (thinking mode disabled)")
                    
            processed_messages.append(message)
            
        return processed_messages 