import re
import sys
import json
import time
import asyncio
import logging
import subprocess
import tiktoken
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("ollamalink")

try:
    cl100k_encoder = tiktoken.get_encoding("cl100k_base") 
except Exception as e:
    logger.warning(f"Failed to load tiktoken encoder: {str(e)}. Falling back to character-based estimation.")
    cl100k_encoder = None

def is_valid_url(url: str) -> bool:
    """Check if a URL is valid."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def load_config():
    """Load configuration from config.json file."""
    config_path = Path(__file__).parent.parent / "config.json"
    default_config = {
        "openai": {
            "api_key": None
        },
        "ollama": {
            "endpoint": "http://localhost:11434"
        },
        "server": {
            "port": 8080,
            "hostname": "127.0.0.1"
        },
        "cloudflared": {
            "use_tunnel": True
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config.json: {str(e)}")
    
    logger.warning("config.json not found, using default configuration")
    return default_config

async def start_cloudflared_tunnel(port, callback=None):
    """
    Start a cloudflared tunnel pointing to localhost.
    
    Args:
        port: The port to expose via the tunnel
        callback: Optional callback function that takes a tunnel URL string when found
                 Used by the GUI version to update the UI
    
    Returns:
        The tunnel URL or None if the tunnel couldn't be started
    """
    try:
        try:
            subprocess.run(
                ["cloudflared", "version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("cloudflared not found or not working")
            return None
        
        process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        url_pattern = re.compile(r'https://[a-zA-Z0-9_.-]+\.trycloudflare\.com')
        tunnel_url = None
        
        logger.info("Starting cloudflared tunnel...")
        
        start_time = time.time()
        while time.time() - start_time < 30:
            line = process.stdout.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
                
            if line.strip():
                logger.info(f"cloudflared: {line.strip()}")
            
            match = url_pattern.search(line)
            if match:
                tunnel_url = match.group(0)
                logger.info(f"Tunnel started at: {tunnel_url}")
                if callback:
                    callback(tunnel_url)
                break
                
            await asyncio.sleep(0.1)
            
        if not tunnel_url:
            logger.error("Could not get tunnel URL within timeout period.")
            return None
            
        return tunnel_url, process
        
    except Exception as e:
        logger.error(f"Error starting tunnel: {str(e)}")
        return None

def is_cloudflared_installed():
    """Check if cloudflared is installed."""
    try:
        subprocess.run(
            ["cloudflared", "version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_cloudflared_install_instructions():
    """Get platform-specific installation instructions for cloudflared."""
    if sys.platform == "darwin":  # macOS
        return "macOS: brew install cloudflare/cloudflare/cloudflared"
    elif sys.platform == "win32":  # Windows
        return "Windows: Download from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"
    else:  # Linux and others
        return "Linux: Download from https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation"

def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken if available, or character-based approximation.
    
    Args:
        text: The text to estimate token count for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
        
    if cl100k_encoder:
        try:
            return len(cl100k_encoder.encode(text))
        except Exception as e:
            logger.debug(f"Error estimating tokens with tiktoken: {str(e)}")
    
    # Fallback to character-based approximation
    return max(1, int(len(text) / 3.5))

def estimate_message_tokens(message: dict) -> int:
    """
    Estimate token count for a chat message using tiktoken when available.
    
    Args:
        message: A message object with 'role' and 'content' keys
        
    Returns:
        Estimated token count including role overhead
    """
    if not message or not isinstance(message, dict):
        return 0
    
    role = message.get("role", "user")
    content = message.get("content", "")
    
    if isinstance(content, list):
        total = 4
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    total += estimate_tokens(item.get("text", ""))
                elif item.get("type") == "image_url":
                    total += 85
            elif isinstance(item, str):
                total += estimate_tokens(item)
        return total + estimate_tokens(role)
    
    if isinstance(content, str):
        if cl100k_encoder:
            try:
                tokens = len(cl100k_encoder.encode(content))
                role_tokens = len(cl100k_encoder.encode(role))
                return tokens + role_tokens + 4
            except Exception:
                pass
                
        return estimate_tokens(content) + 3 + 4
    
    return 10

def count_tokens_in_messages(messages: list) -> int:
    """
    Count tokens in a full list of messages, accounting for the full message formatting.
    
    Args:
        messages: List of message objects with 'role' and 'content' keys
        
    Returns:
        Total token count including all formatting overhead
    """
    if not messages:
        return 0
        
    tokens = sum(estimate_message_tokens(msg) for msg in messages)
    
    return tokens + 3
