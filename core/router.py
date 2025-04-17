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
        self._load_config(config_path)
        self._fetch_models()
        
    def _load_config(self, config_path: str) -> None:
        """Load configuration from config.json."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                if "ollama" in config and "model_mappings" in config["ollama"]:
                    self.model_mappings = config["ollama"]["model_mappings"]
                    logger.info(f"Loaded model mappings from config: {self.model_mappings}")
                    
                    if "default" in self.model_mappings:
                        self.default_model = self.model_mappings["default"]
                        logger.info(f"Set default model from config: {self.default_model}")
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
        try:
            logger.info(f"Fetching models from Ollama at {self.ollama_endpoint}")
            response = requests.get(f"{self.ollama_endpoint}/api/tags", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "models" in data and len(data["models"]) > 0:
                    self.available_models = [m["name"] for m in data["models"] if "name" in m]
                    
                    if not self.default_model and self.available_models:
                        if hasattr(self, 'model_mappings') and "default" in self.model_mappings:
                            default_from_config = self.model_mappings["default"]
                            for model in self.available_models:
                                if (model == default_from_config or 
                                    self._normalize_model_name(model) == self._normalize_model_name(default_from_config)):
                                    self.default_model = model
                                    break
                        
                        if not self.default_model:
                            self.default_model = self.available_models[0]
                    
                    logger.info(f"Default model: {self.default_model}")
                    logger.info(f"Found models: {', '.join(self.available_models)}")
                    return
                else:
                    error_msg = "No models found in Ollama"
                    logger.warning(error_msg)
                    self.connection_error = error_msg
            else:
                error_msg = f"Failed to get models from Ollama: HTTP {response.status_code}"
                logger.warning(error_msg)
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
    
    def get_models_list(self) -> Dict:
        """Get list of models in OpenAI-compatible format."""
        models_list = []
        
        for standard_model, _ in self.model_mappings.items():
            if standard_model != "default":
                models_list.append({
                    "id": standard_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama-proxy"
                })
        
        logger.info(f"Returning {len(models_list)} models to Cursor")
        return {
            "object": "list",
            "data": models_list
        } 