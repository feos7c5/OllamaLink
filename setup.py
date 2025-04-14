from setuptools import setup, find_packages
import sys
import os
from pathlib import Path

# Get the directory where setup.py is located
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Open README.md using an absolute path
readme_path = script_dir / "README.md" 
if not readme_path.exists():
    # Try lowercase name if the uppercase version isn't found
    readme_path = script_dir / "readme.md"

# Read README file or use empty string if file not found
try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    print(f"Warning: Could not find README file at {readme_path}")
    long_description = "OllamaLink - A connector for using local Ollama models in Cursor AI"

# Determine if we're building GUI or CLI
APP_NAME = "OllamaLink"
is_cli = False
if '--cli' in sys.argv:
    is_cli = True
    sys.argv.remove('--cli')
    APP_NAME = "OllamaLink-CLI"

# Additional data files to include
DATA_FILES = [
    ('', ['config.json']),
]

# Common base configuration
setup_args = dict(
    name="ollamalink",
    version="0.1.0",
    author="Python Port",
    author_email="example@example.com", 
    description="A connector for using local Ollama models in Cursor AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/ollamalink",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.23.2",
        "httpx>=0.25.1",
        "pyfiglet>=1.0.2",
        "termcolor>=2.3.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "ollamalink-cli=run_cli:main",
        ],
    },
)


# Handle py2app configuration for macOS
if 'py2app' in sys.argv:
    # Set the main script based on CLI or GUI
    main_script = 'run_cli.py' if is_cli else 'run_gui.py'
    
    # Common py2app options - expanded for better dependency handling
    py2app_options = {
        'argv_emulation': False,  # Set to false to avoid startup issues
        'iconfile': str(script_dir / 'icon.icns') if (script_dir / 'icon.icns').exists() else None,
        'plist': {
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': APP_NAME,
            'CFBundleIdentifier': f'com.example.ollamalink{".cli" if is_cli else ""}',
            'CFBundleVersion': '0.1.0',
            'CFBundleShortVersionString': '0.1.0',
            'NSHumanReadableCopyright': '© 2023 Python Port',
            'NSHighResolutionCapable': True,
            'NSPrincipalClass': 'NSApplication',
            'LSBackgroundOnly': is_cli,  # True for CLI, False for GUI
        },
        'packages': [
            'fastapi', 
            'uvicorn', 
            'httpx', 
            'pyfiglet', 
            'termcolor', 
            'requests',
            'core',
            'PyQt6' if not is_cli else '',
        ],
        'includes': [
            'fastapi.middleware',
            'fastapi.middleware.cors',
            'uvicorn.config',
            'uvicorn.middleware',
            'uvicorn.lifespan',
            'uvicorn.logging',
            'fastapi.applications',
            'starlette.routing',
            'starlette.responses',
            'httpx._config',
            'httpx._models',
            'pkg_resources',
            'pkgutil',
        ],
        'excludes': [
            'tkinter',
            'matplotlib',
            'PyQt5',
        ],
        'resources': [
            'config.json',
        ],
        'site_packages': True,  # Include site-packages for better dependency resolution
    }
    
    # Clean up empty entries
    py2app_options['packages'] = [pkg for pkg in py2app_options['packages'] if pkg]
    
    # Update setup arguments for py2app
    setup_args.update(
        app=[main_script],
        setup_requires=['py2app'],
        options={'py2app': py2app_options},
        data_files=DATA_FILES,
    )

# Run the setup
setup(**setup_args)

# PyInstaller information
if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'pyinstaller-info':
    print("""
To create Windows executables with PyInstaller:

1. Install PyInstaller:
   pip install pyinstaller

2. Build GUI version:
   pyinstaller --name OllamaLink-GUI --onefile --windowed --icon=icon.ico --add-data "config.json;." run_gui.py

3. Build CLI version:
   pyinstaller --name OllamaLink-CLI --onefile --console --icon=icon.ico --add-data "config.json;." run_cli.py

The executables will be created in the dist/ directory.
""")