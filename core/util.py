"""
Utility functions for OllamaLink.
"""
import re
import os
import sys
import json
import time
import asyncio
import logging
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# Set up logger
logger = logging.getLogger("ollamalink")

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
        # Check if cloudflared is installed
        try:
            subprocess.run(
                ["cloudflared", "version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("cloudflared not found or not working")
            return None
        
        # Start the tunnel
        process = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for the tunnel URL
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
