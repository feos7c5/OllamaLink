# OllamaLink Electron GUI

A modern, standalone desktop application for OllamaLink with multi-provider AI support including Ollama and OpenRouter.ai.

## Features

- **Modern Interface**: Sleek, dark-themed UI with intuitive navigation
- **Multi-Provider Support**: Works with both local Ollama and cloud OpenRouter.ai
- **Real-time Monitoring**: Live request/response monitoring and logging
- **Provider Management**: Easy configuration and status monitoring of AI providers
- **Model Browser**: Browse and search available models from all providers
- **Request History**: Detailed request/response history with export capabilities
- **Settings Management**: Persistent settings with import/export functionality
- **Standalone**: Packages into a single executable for easy distribution

## Requirements

- Node.js (v16 or higher)
- Python 3.8+
- Ollama (optional, for local models)
- OpenRouter.ai API key (optional, for cloud models)

## Installation

1. Install dependencies:
```bash
npm install
```

2. Ensure Python dependencies are installed in the parent directory:
```bash
cd ..
pip install -r requirements.txt  # If available
```

## Development

Start the application in development mode:
```bash
npm run dev
```

## Building

### Build for current platform:
```bash
npm run build
```

### Build for specific platforms:
```bash
npm run build-win    # Windows
npm run build-mac    # macOS
npm run build-linux  # Linux
```

### Create unpacked directory (for testing):
```bash
npm run pack
```

## Configuration

The application will automatically:
1. Start the Python backend on an available port
2. Load configuration from the parent directory's `config.json`
3. Provide a UI for setting API keys and other preferences

### API Keys

Configure your API keys in the Settings tab:
- **OpenRouter API Key**: For accessing OpenRouter.ai models
- **OpenAI API Key**: For direct OpenAI integration (if configured)

## Architecture

- **Frontend**: Electron + Modern HTML/CSS/JavaScript
- **Backend**: Python FastAPI server (from parent directory)
- **Communication**: HTTP API calls between Electron and Python
- **Data**: Persistent settings stored via electron-store

## Tabs Overview

- **Dashboard**: Server control and status overview
- **Providers**: Provider configuration and health monitoring  
- **Models**: Browse available models from all providers
- **Requests**: Request history and detailed inspection
- **Console**: Real-time logs and debugging information
- **Settings**: Application configuration and API key management

## Directory Structure

```
electron-gui/
├── src/
│   ├── main.js          # Electron main process
│   ├── preload.js       # Secure API bridge
│   └── renderer/        # Frontend application
│       ├── index.html   # Main UI
│       ├── styles.css   # Modern styling
│       └── app.js       # Application logic
├── assets/              # Icons and resources
├── package.json         # Dependencies and build config
└── README.md           # This file
```

## Security

- Context isolation enabled
- Node integration disabled in renderer
- Secure API bridge via preload script
- Sensitive data stored securely via electron-store

## Troubleshooting

1. **Backend won't start**: Ensure Python dependencies are installed and the parent directory contains the OllamaLink core modules.

2. **Port conflicts**: The app automatically finds available ports, but you can check the console for port information.

3. **Provider errors**: Check the Console tab for detailed error messages and ensure API keys are configured correctly.

4. **Build issues**: Ensure all dependencies are installed and you have the required build tools for your platform.

## License

Same as parent OllamaLink project.
