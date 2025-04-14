import asyncio
import argparse
import logging
from pathlib import Path

import uvicorn
import termcolor
from pyfiglet import Figlet

from core.api import create_api
from core.router import OllamaRouter
from core.util import load_config, start_cloudflared_tunnel, is_cloudflared_installed, get_cloudflared_install_instructions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ollamalink")

async def start_cloudflared_tunnel_cli(port):
    """
    Start a cloudflared tunnel and handle CLI-specific output.
    
    This is a wrapper around the core utility function that adds
    CLI-specific colored output.
    """
    if not is_cloudflared_installed():
        print(termcolor.colored("cloudflared not found. Install instructions:", 'red'))
        print(get_cloudflared_install_instructions())
        return None

    print(termcolor.colored("Starting cloudflared tunnel...", 'yellow'))
    
    result = await start_cloudflared_tunnel(port)
    
    if not result:
        print(termcolor.colored("Could not get tunnel URL. Check output above.", 'red'))
        return None
    
    tunnel_url, process = result
    print(termcolor.colored(f"Tunnel started at: {tunnel_url}", 'green'))
    return tunnel_url

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
                        default=config["cloudflared"]["use_tunnel"],
                        help="Use cloudflared tunnel (default: on)")
    
    parser.add_argument("--no-tunnel", dest="tunnel", action="store_false",
                        help="Disable cloudflared tunnel")
    
    args = parser.parse_args()
    
    # --direct is an alias for --no-tunnel
    if args.direct:
        args.tunnel = False
    
    f = Figlet(font='slant')
    print(termcolor.colored(f.renderText('OllamaLink'), 'green'))
    print(termcolor.colored("Connect Cursor with Ollama models\n", 'cyan'))
    
    print(termcolor.colored("Configuration:", 'yellow'))
    print(f"• Ollama endpoint: {args.ollama}")
    print(f"• Server port: {args.port}")
    print(f"• Server host: {args.host}")
    print(f"• Using tunnel: {'Yes' if args.tunnel else 'No'}")
    print()
    
    router = OllamaRouter(ollama_endpoint=args.ollama)
    
    if hasattr(router, 'connection_error') and router.connection_error:
        print(termcolor.colored(f"Error: {router.connection_error}", 'red'))
        print(termcolor.colored("OllamaLink will still start, but it won't be able to use models from Ollama", 'yellow'))
        print(termcolor.colored("Troubleshooting:", 'yellow'))
        print("• Make sure Ollama is running: ollama serve")
        print(f"• Verify Ollama API is accessible: {args.ollama}/api/tags")
        print("• Check for network/firewall issues if using a remote Ollama instance")
        print(termcolor.colored("\nOllamaLink will use fallback settings until Ollama is available", 'yellow'))
    elif router.available_models:
        print(termcolor.colored(f"Found {len(router.available_models)} models:", 'green'))
        for model in router.available_models:
            print(f"• {model}")
        
        print(termcolor.colored(f"\nDefault model: {router.default_model}", 'green'))
    else:
        print(termcolor.colored("No Ollama models found. Is Ollama running?", 'red'))
        print(termcolor.colored("Please make sure Ollama is running with: ollama serve", 'yellow'))
    
    print()
    print(termcolor.colored("Cursor will map standard model requests to:", 'yellow'))
    
    for api_model, local_model in router.model_mappings.items():
        if api_model != "default":
            resolved_model = router.get_model_name(api_model)
            print(f"• {api_model} → {resolved_model}")
    
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
            
            tunnel_url = await start_cloudflared_tunnel_cli(args.port)
            
            if tunnel_url:
                cursor_url = f"{tunnel_url}/v1"
                
                print("\n" + "=" * 60)
                print(termcolor.colored("\nReady! Configure Cursor with these settings:", 'green'))
                print(termcolor.colored(f"\n  URL: {cursor_url}", 'white', attrs=['bold']))
                print("\nIn Cursor settings > AI > Override OpenAI Base URL")
                print("\nThen make sure to select one of mapped models in Cursor:")
                for model in router.model_mappings.keys():
                    if model != "default":
                        print(f"- {model}")
                print("=" * 60 + "\n")
                
                print(termcolor.colored("OllamaLink is running. Press Ctrl+C to stop.", 'cyan'))
            
            await server_task
            
        try:
            asyncio.run(run_with_tunnel())
        except KeyboardInterrupt:
            print(termcolor.colored("\nShutting down...", 'yellow'))
    
    else:
        base_url = f"http://{args.host}:{args.port}/v1"
        
        print("\n" + "=" * 60)
        print(termcolor.colored("\nReady! Configure Cursor with these settings:", 'green'))
        print(termcolor.colored(f"\n  URL: {base_url}", 'white', attrs=['bold']))
        print("\nIn Cursor settings > AI > Override OpenAI Base URL")
        print("=" * 60 + "\n")
        
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
