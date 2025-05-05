import re
import json
import asyncio
import logging
import subprocess
import tiktoken
from pathlib import Path
from urllib.parse import urlparse
import platform
from typing import Dict, Any, Optional, Tuple, List

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

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.json with fallbacks to defaults.
    """
    try:
        config_path = Path("config.json")
        
        if not config_path.exists():
            default_config = {
                "server": {
                    "port": 8080,
                    "hostname": "127.0.0.1"
                },
                "ollama": {
                    "endpoint": "http://localhost:11434",
                    "thinking_mode": True,
                    "skip_integrity_check": False,
                    "max_streaming_tokens": 32000,
                    "model_mappings": {
                        "default": "llama3",
                        "gpt-3.5-turbo": "llama3",
                        "gpt-4": "llama3",
                        "gpt-4o": "llama3"
                    }
                },
                "tunnel": {
                    "use_tunnel": True,
                    "type": "localhost_run"
                }
            }
            
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=4)
                
            return default_config
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        # Ensure the required sections exist
        if "server" not in config:
            config["server"] = {"port": 8080, "hostname": "127.0.0.1"}
            
        if "ollama" not in config:
            config["ollama"] = {"endpoint": "http://localhost:11434", "model_mappings": {"default": "llama3"}}
        
        if "model_mappings" not in config["ollama"]:
            config["ollama"]["model_mappings"] = {"default": "llama3"}
            
        if "tunnel" not in config:
            config["tunnel"] = {"use_tunnel": True, "type": "localhost_run"}
            
        if "default" not in config["ollama"]["model_mappings"]:
            config["ollama"]["model_mappings"]["default"] = "llama3"
            
        return config
    
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        
        return {
            "server": {"port": 8080, "hostname": "127.0.0.1"},
            "ollama": {
                "endpoint": "http://localhost:11434",
                "model_mappings": {"default": "llama3"}
            },
            "tunnel": {"use_tunnel": True, "type": "localhost_run"}
        }

async def start_localhost_run_tunnel(port: int, callback=None) -> Optional[Tuple[str, Any]]:
    """
    Start a localhost.run tunnel using SSH and return the URL and process.
    
    Args:
        port: The local port to tunnel to
        callback: Optional callback function to receive the tunnel URL
        
    Returns:
        Optional[Tuple[str, Any]]: A tuple of (tunnel_url, process) if successful,
                                  or None if the tunnel couldn't be started
    """
    logger.info("Starting localhost.run tunnel...")
    
    # Check if another instance of the tunnel might already be running
    try:
        if platform.system() == "Windows":
            check_process = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ssh.exe"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            if "localhost.run" in check_process.stdout:
                termination_cmd = f'taskkill /F /FI "WINDOWTITLE eq localhost.run" /T'
                try:
                    subprocess.run(termination_cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except:
                    pass 
        else:
            check_process = subprocess.run(
                ["ps", "aux"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            if "localhost.run" in check_process.stdout:
                try:
                    subprocess.run("pkill -f 'ssh.*localhost.run'", shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                except:
                    pass
    except Exception as e:
        logger.debug(f"Could not check for existing tunnel processes: {str(e)}")
    
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ["where", "ssh"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            result = subprocess.run(
                ["which", "ssh"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
        if result.returncode != 0:
            logger.error("SSH is not installed or not found in PATH")
            return None
        
        # Start SSH tunnel to localhost.run
        # Use -R 80:localhost:port to create a tunnel from the 
        # localhost.run server to our local server on the specified port
        process = await asyncio.create_subprocess_exec(
            "ssh", 
            "-o", "ServerAliveInterval=60", 
            "-o", "ServerAliveCountMax=60",
            "-o", "StrictHostKeyChecking=no", 
            "-o", "UserKnownHostsFile=/dev/null", 
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"80:localhost:{port}", 
            "nokey@localhost.run", 
            "--",
            "--inject-http-proxy-headers", 
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )
        
        tunnel_url = None
        start_time = asyncio.get_event_loop().time()
        timeout = 120 
        
        logger.info("Waiting for localhost.run tunnel to start...")
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if process.stdout:
                line = await process.stdout.readline()
                if not line:
                    if process.returncode is not None:
                        logger.error(f"localhost.run tunnel exited with code {process.returncode}")
                        return None
                    continue
                    
                line_str = line.decode('utf-8', errors='ignore')
                logger.debug(f"localhost.run: {line_str.strip()}")
                
                if "admin.localhost.run" in line_str:
                    logger.debug("Ignoring admin.localhost.run URL - this is not a valid tunnel URL")
                    continue
                
                # Several detection patterns, in order of reliability:
                
                # Pattern 1: Line contains "tunneled with tls termination" - used by free tier
                if "tunneled with tls termination" in line_str:
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (TLS termination): {tunnel_url}")
                        
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 2: Line containing "is forwarding to localhost:port"
                elif f"forwarding to localhost:{port}" in line_str:
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (forwarding): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 3: Line contains "Follow" - typically "Follow this link:"
                elif "Follow" in line_str:
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (follow link): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 4: Line contains the word "your connection" with hostname pattern
                elif "your connection" in line_str.lower():
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (connection info): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 5: Line contains "https://" and ".lhr.life" 
                elif "https://" in line_str and ".lhr.life" in line_str:
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.lhr\.life', line_str)
                    if match:
                        found_url = match.group(0)
                        tunnel_url = found_url
                        logger.info(f"Tunnel URL found (lhr.life domain): {tunnel_url}")
                        
                        if callback:
                            callback(tunnel_url)
                            
                        return tunnel_url, process
                
                # Pattern 6: Line contains "https://" and ".localhost.run" but not "admin.localhost.run"
                elif "https://" in line_str and ".localhost.run" in line_str and "admin.localhost.run" not in line_str:
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.localhost\.run', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (localhost.run domain): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 7: Line contains "tunneled through" - common in nokey output
                elif "tunneled through" in line_str.lower():
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (tunneled through): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 8: Numeric subdomain pattern - fallback
                elif re.search(r'\d+\.(lhr\.life|localhost\.run)', line_str):
                    match = re.search(r'https?://\d+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (numeric pattern): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                # Pattern 9: General URL detection - fallback
                elif re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str):
                    match = re.search(r'https?://[a-zA-Z0-9.-]+\.(lhr\.life|localhost\.run)', line_str)
                    if match:
                        found_url = match.group(0)
                        if "admin.localhost.run" not in found_url:
                            tunnel_url = found_url
                            logger.info(f"Tunnel URL found (general URL): {tunnel_url}")
                            
                            if callback:
                                callback(tunnel_url)
                                
                            return tunnel_url, process
                
                if "permission denied" in line_str.lower():
                    logger.debug(f"SSH permission denied. Already using 'nokey@localhost.run'")
                    if "publickey" in line_str.lower():
                        logger.debug("SSH keys are being ignored. Using direct connection.")
                    
                if "connection refused" in line_str.lower():
                    logger.error(f"Connection refused to localhost.run")
                    
                if "no route to host" in line_str.lower():
                    logger.error(f"No route to localhost.run - check your internet connection")
            
            await asyncio.sleep(0.1)
        
        logger.error("Timed out waiting for localhost.run tunnel URL")
        return None
        
    except Exception as e:
        logger.error(f"Error starting localhost.run tunnel: {str(e)}")
        return None

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
    
    return max(1, int(len(text) / 3.5))

def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """
    Roughly estimate the number of tokens in a message.
    This is a basic approximation, not a precise count.
    """
    if not message or "content" not in message:
        return 0
    
    content = message["content"]
    role = message.get("role", "user")
    
    if not content:
        return 0
    
    if isinstance(content, str):
        char_count = len(content)
        token_estimate = char_count / 4
    
        return int(token_estimate) + 5

    elif isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    total += len(item["text"]) / 4
                elif item.get("type") == "image_url" and "image_url" in item:
                    total += 50
            elif isinstance(item, str):
                total += len(item) / 4
        return int(total) + 5
    
    return 5

def count_tokens_in_messages(messages: List[Dict[str, Any]]) -> int:
    """
    Count the approximate number of tokens in a list of messages.
    This is a rough estimation for practical purposes.
    """
    return sum(estimate_message_tokens(message) for message in messages)
