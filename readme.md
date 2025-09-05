# OllamaLink

**Connect Cursor AI to your local Ollama models or OpenRouter.ai**

OllamaLink is a bridge application that connects Cursor AI to your local Ollama models or cloud-based OpenRouter.ai models. It supports hybrid model access, allowing you to use both local and cloud models seamlessly. Currently, only the CLI version is functional with full Ollama and OpenRouter integration. The GUI is not yet available.

## Features

- **Hybrid Model Access**: Connect to both local Ollama models and cloud OpenRouter.ai models
- **Simple**: Minimal setup, just run and go
- **Private**: Your code stays on your machine with local models
- **Flexible**: Use any Ollama or OpenRouter model with Cursor
- **Intelligent Routing**: Automatic provider selection and fallback mechanisms
- **Tunnel**: Works with Cursor's cloud service via localhost.run tunnels
- **CLI Only**: Currently only command-line interface is available with full functionality
- **GUI**: Graphical interface is in development (not yet available)
- **Model Mapping**: Map custom model names to your local Ollama or cloud models
- **Real-time Monitoring**: Track requests and responses in the GUI
- **Secure**: Optional API key support for added security
- **No Timeout Limits**: Long-running operations now supported without constraints

## Quick Start

1. **Install Ollama**:
   Ensure you have [Ollama](https://ollama.ai/) installed and running:
   ```sh
   ollama serve
   ```

2. **Install Requirements**:
   Install the necessary dependencies using pip:
   ```sh
   pip install -r requirements.txt
   # Or using the modern pyproject.toml:
   pip install -e .
   ```

3. **Run OllamaLink**:
   Currently only the CLI version is available:

   **CLI Version** (Full Ollama + OpenRouter support):
   ```sh
   python run_cli.py
   ```
   
   **Note**: GUI version is not yet functional and is currently in development.

## Setting Up Cursor

### 1. Install Cursor

First, download and install Cursor from the official website: [https://cursor.sh](https://cursor.sh)

### 2. Get the OllamaLink URL

When you start OllamaLink, you'll get two types of URLs:

1. **Local URL**: `http://localhost:8080/v1` (default)
2. **Tunnel URL**: A public URL like `https://randomsubdomain.localhost.run/v1`

Choose the appropriate URL based on your needs:
- Use the **Local URL** if running Cursor on the same machine and port forwarding at router
- Use the **Tunnel URL** if you want to access your models from anywhere (recommended)

### 3. Configure Cursor Settings

1. Open Cursor and access settings:
   - On macOS: Press `⌘ + ,` (Command + Comma)
   - On Windows/Linux: Press `Ctrl + ,`
   - Or click on the gear icon in the bottom left corner

2. Navigate to the "Models" tab in settings

3. Configure the following settings:
   - Find "Override OpenAI Base URL" below point OpenAI API Key
   - Paste your OllamaLink URL (either local or tunnel URL) and press save
   - Make sure to include the `/v1` at the end of the URL
   - Past API Key if specified in config.json or let it empty
   - Press Verify behind key input and it should automatically detect and start

4. Select a Model:
   Ensure the mapped models are availiable in cursor. 
   - If not add them e.g. qwen2.5": "qwen2.5" in config you need to daa qwen2.5 as model in cursor

   The actual Ollama model used depends on your `config.json` mappings.

5. Test the Connection:
   - Click the "Test Connection" button
   - You should see a success message
   - If you get an error, check the troubleshooting section below

### Important Note on Model Names

When setting up model mappings in `config.json`, please note:

- You **cannot** use existing commercial model names like "gpt-4o" or "claude-3.5" as the model names in Cursor

### Common Setup Issues

1. **URL Error**:
   - Make sure the URL ends with `/v1`
   - Check if OllamaLink is running
   - Try the local URL if tunnel isn't working

2. **Model Not Found**:
   - Ensure you've selected one of the supported model names
   - Check your model mappings in `config.json`
   - Verify your Ollama models with `ollama list`

3. **Connection Failed**:
   - Verify Ollama is running with `ollama serve`
   - Check OllamaLink logs for errors
   - Try restarting both OllamaLink and Cursor

4. **Tunnel Issues**:
   - Ensure SSH is installed on your system for localhost.run tunnels
   - Check the console logs for any tunnel connection errors
   - If you see "permission denied" errors, make sure your SSH setup is correct
   - The system will check for existing tunnels to avoid conflicts

## Configuration

You can customize OllamaLink using a `config.json` file in the project root:

```json
{
    "openai": {
        "api_key": "sk-proj-1234567890",
        "endpoint": "https://api.openai.com"
    },
    "ollama": {
        "endpoint": "http://localhost:11434",
        "model_mappings": {
            "gpt-4o": "qwen2",
            "gpt-3.5-turbo": "llama3", 
            "claude-3-opus": "wizardcoder",
            "default": "qwen2.5-coder"
        },
        "thinking_mode": true,
        "skip_integrity_check": true,
        "max_streaming_tokens": 32000
    },
    "openrouter": {
        "enabled": false,
        "api_key": "sk-or-1234567890",
        "endpoint": "https://openrouter.ai/api/v1",
        "model_mappings": {
            "gpt-4o": "openai/gpt-4o",
            "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
            "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct"
        },
        "priority": 2,
        "fallback_enabled": true
    },
    "routing": {
        "provider_priority": ["ollama", "openrouter"],
        "fallback_enabled": true,
        "health_check_interval": 30
    },
    "server": {
        "port": 8080,
        "hostname": "127.0.0.1"
    },
    "tunnels": {
        "use_tunnel": true,
        "preferred": "localhost.run"
    }
}
```

## OpenRouter Setup (CLI Only)

To enable OpenRouter.ai cloud models:

1. **Get an OpenRouter API Key**:
   - Visit [OpenRouter.ai](https://openrouter.ai/) and create an account
   - Generate an API key from your dashboard

2. **Configure OpenRouter in config.json**:
   ```json
   "openrouter": {
       "enabled": true,
       "api_key": "sk-or-your-api-key-here",
       "model_mappings": {
           "gpt-4o": "openai/gpt-4o",
           "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet"
       }
   }
   ```

3. **Run with CLI**:
   ```sh
   python run_cli.py
   ```
   The system will automatically route requests between local Ollama and OpenRouter based on model availability and your configured priorities.

## GUI Status

⚠️ **The GUI is currently not functional and is in development.** 

Planned GUI features (not yet available):
- Dashboard for server status and model mappings
- Console for real-time server logs and events
- Request/Response monitoring
- Settings configuration

For now, please use the CLI version which has all functionality implemented.

## CLI Usage

```sh
python run_cli.py [options]
```

### Options:
- `--port PORT`: Port to run on (default from config.json or 8080)
- `--direct`: Direct mode without tunnel
- `--ollama URL`: Ollama API URL (default from config.json or http://localhost:11434)
- `--host HOST`: Host to bind to (default from config.json or 127.0.0.1)
- `--tunnel`: Use localhost.run tunnel (default: on)
- `--no-tunnel`: Disable tunnel

## Model Mapping

OllamaLink provides flexible model mapping that allows you to route requests for commercial models (like GPT-4 or Claude) to your locally running Ollama models.

### How Model Mapping Works

1. **Direct Mapping**: Each entry maps a client model name to a local model pattern.
   
   For example, when a client requests `gpt-4o`, OllamaLink will:
   - Look for a model that exactly matches "qwen2"
   - If not found, look for a model that contains "qwen2" in its name
   - If still not found, fall back to the default model

2. **Default Model**: The `default` entry specifies which model to use when no appropriate mapping is found.

3. **Fuzzy Matching**: The router performs fuzzy matching, so "qwen2" will match models like "qwen2.5-coder:latest", "qwen2-7b-instruct", etc.

## Building Executables

### macOS App Bundle

Build using py2app (CLI only, GUI not yet available):

```sh
# CLI Version
python setup.py py2app --cli
```

### Windows Executable

Build using PyInstaller (CLI only, GUI not yet available):

```sh
# CLI Version
pyinstaller --name OllamaLink-CLI --onefile --console --icon=icon.ico --add-data "config.json;." run_cli.py
```

## Troubleshooting

### Cannot Connect to Ollama

If OllamaLink starts but shows an error connecting to Ollama:

1. **Check if Ollama is Running**:
   ```sh
   ollama serve
   ```

2. **Verify API Access**:
   Open your browser to: `http://localhost:11434/api/tags`
   You should see a JSON response with your models.

3. **Ensure at Least One Model is Installed**:
   ```sh
   ollama list
   ```
   If no models are shown, install one:
   ```sh
   ollama pull qwen2.5-coder
   ```

4. **Connection Issues**:
   - Check your firewall settings if using a remote Ollama server
   - Verify the Ollama endpoint in config.json if you changed it from default

### "No completion was generated after max retries"

If you encounter this error in Cursor:
```
We encountered an issue when using your API key: No completion was generated after max retries
API Error: Unknown error
(Request ID: xxxx-xxxx-xxxx-xxxx)
```

Try these steps:
1. **Ensure Ollama Models**:
   Make sure at least one model is loaded in Ollama:
   ```sh
   ollama list
   ```
   If no models are listed, run: `ollama pull qwen2.5-coder`

2. **Restart OllamaLink**:
   ```sh
   python run.py
   ```

3. **Update Cursor Model Selection**:
   In Cursor, make sure you're using mapped model names

## Recent Changes

### OpenRouter.ai Integration (CLI)
- **NEW**: Full OpenRouter.ai cloud model support in CLI version
- Hybrid model access with intelligent routing between local Ollama and cloud OpenRouter
- Automatic provider fallback when primary provider is unavailable
- Cost optimization with configurable provider priority (local-first by default)
- Support for all major OpenRouter models (GPT-4, Claude, Llama, etc.)
- Real-time provider health monitoring
- **Note**: GUI support for OpenRouter is currently in development

### Tunneling Improvements
- Switched to localhost.run for more reliable connections
- Enhanced tunnel URL detection for various connection scenarios
- Added checks for existing tunnel processes to prevent conflicts
- Improved error handling and logging for tunnel connections

### Thinking Mode
- Added configurable thinking mode to control how models generate responses
- When enabled (default), models perform thorough analysis before responding
- Can be disabled with the `thinking_mode: false` setting for faster, more direct responses
- Automatically adds the `/no_think` prefix to user messages when disabled

### Timeout Constraints Removed
- Removed all artificial timeout limitations
- Support for long-running operations without time constraints
- Fixed handling of structured message content for compatibility with Ollama's API
- Improved model listing and availability detection

## License

MIT
