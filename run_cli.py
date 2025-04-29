import asyncio
import argparse
import logging
from pathlib import Path
import json

import uvicorn
import termcolor
from pyfiglet import Figlet

from core.api import create_api
from core.router import OllamaRouter
from core.util import load_config, start_localhost_run_tunnel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ollamalink")

HEADER_COLOR = 'green'
SUBHEADER_COLOR = 'cyan'
INFO_COLOR = 'yellow'
ERROR_COLOR = 'red'
SUCCESS_COLOR = 'green'
CODE_COLOR = 'cyan'
DIVIDER = "─" * 60

async def start_tunnel_cli(port):
    """
    Start a localhost.run tunnel and handle CLI-specific output.
    
    This is a wrapper around the core utility function that adds
    CLI-specific colored output.
    """
    print(termcolor.colored("🔄 Starting localhost.run tunnel...", INFO_COLOR))
    
    result = await start_localhost_run_tunnel(port)
    
    if not result:
        print(termcolor.colored("❌ Could not get tunnel URL. Check output above.", ERROR_COLOR))
        return None
    
    tunnel_url, process = result
    print(termcolor.colored(f"✅ Tunnel started at: {tunnel_url}", SUCCESS_COLOR))
    return tunnel_url

def display_model_error(error_message, error_type):
    """
    Display user-friendly error messages for model-related issues.
    
    Args:
        error_message: The error message to display
        error_type: The type of error (model_not_found, model_corrupted, etc.)
    """
    print(f"\n{DIVIDER}")
    print(termcolor.colored("❌ Model Error Detected", ERROR_COLOR, attrs=['bold']))
    print(f"{DIVIDER}\n")
    
    if error_type == "model_corrupted":
        print(termcolor.colored("One or more Ollama model files appear to be corrupted.", ERROR_COLOR))
        print()
        print(termcolor.colored("📋 Troubleshooting Steps:", INFO_COLOR, attrs=['bold']))
        
        print("1. First, try pulling the model again to repair it:")
        print(termcolor.colored(f"   ollama pull MODEL_NAME", CODE_COLOR))
        print()
        
        print("2. If that doesn't work, remove the model completely and reinstall:")
        print(termcolor.colored(f"   ollama rm MODEL_NAME", CODE_COLOR))
        print(termcolor.colored(f"   ollama pull MODEL_NAME", CODE_COLOR))
        print()
        
        print("3. If problems persist, you may need to restart Ollama or check disk space")
        print()
        
        print(termcolor.colored("ℹ️  Error details:", INFO_COLOR))
        print(f"   {error_message}")
    
    elif error_type == "model_not_found":
        print(termcolor.colored("The requested model was not found in Ollama.", ERROR_COLOR))
        print()
        print(termcolor.colored("📋 Available Commands:", INFO_COLOR, attrs=['bold']))
        
        print("• List available models:")
        print(termcolor.colored("   ollama list", CODE_COLOR))
        print()
        
        print("• Pull a new model:")
        print(termcolor.colored("   ollama pull MODEL_NAME", CODE_COLOR))
        print()
        
        print(termcolor.colored("ℹ️  Error details:", INFO_COLOR))
        print(f"   {error_message}")
    
    elif error_type == "connection_error":
        print(termcolor.colored("Cannot connect to Ollama.", ERROR_COLOR))
        print()
        print(termcolor.colored("📋 Troubleshooting:", INFO_COLOR, attrs=['bold']))
        
        print("1. Make sure Ollama is running:")
        print(termcolor.colored("   ollama serve", CODE_COLOR))
        print()
        
        print("2. Check the Ollama logs for errors")
        print()
        
        print(termcolor.colored("ℹ️  Error details:", INFO_COLOR))
        print(f"   {error_message}")
    
    else:
        print(termcolor.colored(f"Error: {error_message}", ERROR_COLOR))
    
    print(f"\n{DIVIDER}\n")

def main():
    """Run the OllamaLink server."""
    config = load_config()
    
    parser = argparse.ArgumentParser(description="OllamaLink - Connect Cursor AI to Ollama models")
    
    parser.add_argument("--port", "-p", type=int, 
                        default=config["server"]["port"],
                        help=f"Port to run the server on (default: {config['server']['port']})")
    
    parser.add_argument("--host", "-H", type=str, 
                        default=config["server"]["hostname"],
                        help=f"Host to bind the server to (default: {config['server']['hostname']})")
    
    parser.add_argument("--ollama", "-o", type=str, 
                        default=config["ollama"]["endpoint"],
                        help=f"Ollama API endpoint (default: {config['ollama']['endpoint']})")
    
    parser.add_argument("--direct", "-d", action="store_true",
                        help="Direct mode (no tunnel)")
    
    parser.add_argument("--tunnel", "-t", action="store_true", 
                        default=config["tunnel"]["use_tunnel"],
                        help="Use localhost.run tunnel (default: on)")
    
    parser.add_argument("--no-tunnel", dest="tunnel", action="store_false",
                        help="Disable tunnel")
    
    parser.add_argument("--thinking", dest="thinking_mode", action="store_true",
                        default=config["ollama"].get("thinking_mode", True),
                        help="Enable thinking mode (default)")
    
    parser.add_argument("--no-thinking", dest="thinking_mode", action="store_false",
                        help="Disable thinking mode (adds /no_think prefix to prompts)")
    
    parser.add_argument("--skip-integrity", dest="skip_integrity", action="store_true",
                        default=config["ollama"].get("skip_integrity_check", False),
                        help="Skip model integrity checks (helps with timeout errors)")
    
    parser.add_argument("--check-integrity", dest="skip_integrity", action="store_false",
                        help="Perform model integrity checks")
    
    parser.add_argument("--max-tokens", type=int,
                        default=config["ollama"].get("max_streaming_tokens", 32000),
                        help="Maximum token limit for streaming requests (default: 32000)")
    
    args = parser.parse_args()
    
    if args.direct:
        args.tunnel = False
    
    f = Figlet(font='slant')
    print(termcolor.colored(f.renderText('OllamaLink'), HEADER_COLOR))
    print(termcolor.colored("Connect Cursor with Ollama models\n", SUBHEADER_COLOR))
    
    print(termcolor.colored("⚙️  Configuration:", INFO_COLOR, attrs=['bold']))
    print(f"• Ollama endpoint: {args.ollama}")
    print(f"• Server port: {args.port}")
    print(f"• Server host: {args.host}")
    print(f"• Using tunnel: {'Yes' if args.tunnel else 'No'}")
    print(f"• Thinking mode: {'Enabled' if args.thinking_mode else 'Disabled (using /no_think)'}")
    print(f"• Integrity checks: {'Disabled' if args.skip_integrity else 'Enabled'}")
    print(f"• Max streaming tokens: {args.max_tokens}")
    print()
    
    router = OllamaRouter(ollama_endpoint=args.ollama)
    router.thinking_mode = args.thinking_mode
    router.skip_integrity_check = args.skip_integrity
    
    if hasattr(router, 'connection_error') and router.connection_error:
        display_model_error(router.connection_error, "connection_error")
        print(termcolor.colored("⚠️  OllamaLink will still start, but it won't be able to use models from Ollama", INFO_COLOR))
        print(termcolor.colored("OllamaLink will use fallback settings until Ollama is available", INFO_COLOR))
    elif hasattr(router, 'model_error') and router.model_error:
        display_model_error(
            router.model_error.get('message', 'Unknown model error'), 
            router.model_error.get('type', 'unknown')
        )
        print(termcolor.colored("⚠️  OllamaLink will still start with available models", INFO_COLOR))
    elif router.available_models:
        print(termcolor.colored(f"✅ Found {len(router.available_models)} models:", SUCCESS_COLOR, attrs=['bold']))
        for model in router.available_models:
            print(f"• {model}")
        
        print(termcolor.colored(f"\n🎯 Default model: {router.default_model}", SUCCESS_COLOR, attrs=['bold']))
    else:
        print(termcolor.colored("❌ No Ollama models found. Is Ollama running?", ERROR_COLOR))
        print(termcolor.colored("Please make sure Ollama is running with: ollama serve", INFO_COLOR))
    
    print()
    print(termcolor.colored("🔄 Cursor will map standard model requests to:", INFO_COLOR, attrs=['bold']))
    
    for api_model, local_model in router.model_mappings.items():
        if api_model != "default":
            resolved_model = router.get_model_name(api_model)
            print(f"• {api_model} → {resolved_model}")
    
    if args.max_tokens != config["ollama"].get("max_streaming_tokens", 32000):
        config["ollama"]["max_streaming_tokens"] = args.max_tokens
        try:
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            print(termcolor.colored(f"ℹ️  Updated max_streaming_tokens in config.json to {args.max_tokens}", INFO_COLOR))
        except Exception as e:
            print(termcolor.colored(f"⚠️  Could not update config.json: {str(e)}", ERROR_COLOR))
    
    app = create_api(ollama_endpoint=args.ollama)
    
    if args.tunnel:
        async def run_with_tunnel():
            config = uvicorn.Config(
                app, 
                host="0.0.0.0", 
                port=args.port,
                log_level="warning"
            )
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())
            
            await asyncio.sleep(1)
            
            tunnel_url = await start_tunnel_cli(args.port)
            
            if tunnel_url:
                cursor_url = f"{tunnel_url}/v1"
                print(f"\n{DIVIDER}")
                print(termcolor.colored(f"✅ OllamaLink server is running!", SUCCESS_COLOR, attrs=['bold']))
                print(f"{DIVIDER}\n")
                print(termcolor.colored("🔗 Use this URL in Cursor AI:", INFO_COLOR, attrs=['bold']))
                print(termcolor.colored(f"{cursor_url}", CODE_COLOR, attrs=['bold']))
                print()
                print(termcolor.colored("📝 Instructions:", INFO_COLOR, attrs=['bold']))
                print("1. In Cursor, go to settings > AI > Configure AI Provider")
                print("2. Choose OpenAI Compatible and paste the URL above")
                print("3. Chat with local Ollama models in Cursor!")
                print()
                print(termcolor.colored("Press CTRL+C to stop the server", INFO_COLOR))
                print()
                
            await server_task
                
        asyncio.run(run_with_tunnel())
    else:
        if args.host == "0.0.0.0":
            host_display = "localhost"
        else:
            host_display = args.host
            
        local_url = f"http://{host_display}:{args.port}/v1"
        
        print(f"\n{DIVIDER}")
        print(termcolor.colored(f"✅ OllamaLink server is running!", SUCCESS_COLOR, attrs=['bold']))
        print(f"{DIVIDER}\n")
        print(termcolor.colored("🔗 Use this URL in Cursor AI:", INFO_COLOR, attrs=['bold']))
        print(termcolor.colored(f"{local_url}", CODE_COLOR, attrs=['bold']))
        print()
        print(termcolor.colored("⚠️  Important:", ERROR_COLOR, attrs=['bold']))
        print("You're running in direct mode without a tunnel.")
        print("Cursor AI must be running on the same machine as OllamaLink.")
        print()
        print(termcolor.colored("📝 Instructions:", INFO_COLOR, attrs=['bold']))
        print("1. In Cursor, go to settings > AI > Configure AI Provider")
        print("2. Choose OpenAI Compatible and paste the URL above")
        print("3. Chat with local Ollama models in Cursor!")
        print()
        print(termcolor.colored("Press CTRL+C to stop the server", INFO_COLOR))
        print()

        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
