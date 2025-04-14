@echo off
REM Build script for Windows using PyInstaller

REM Find the script directory (where build_win.bat is located)
SET SCRIPT_DIR=%~dp0
CD %SCRIPT_DIR%

REM Detect Python command
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=python3
    ) else (
        echo Error: Python not found. Please install Python 3.
        exit /b 1
    )
)

echo Using Python command: %PYTHON_CMD%

REM Create dist directory in root if it doesn't exist
if not exist dist mkdir dist

REM Install required dependencies
echo Installing required dependencies...
%PYTHON_CMD% -m pip install pyinstaller
%PYTHON_CMD% -m pip install -r requirements.txt

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
mkdir dist

REM Process arguments
if "%1"=="cli" (
    echo Building CLI version...
    %PYTHON_CMD% -m PyInstaller --name OllamaLink-CLI --onefile --console --distpath=dist --workpath=build --icon=icon.ico --add-data "config.json;." run_cli.py
    
    if not exist dist\OllamaLink-CLI.exe (
        echo Error: CLI build failed - executable not created
        exit /b 1
    )
    
    echo Build complete! The CLI executable is available in the dist\ directory.
) else if "%1"=="all" (
    echo Building both GUI and CLI versions...
    
    REM Build GUI version
    echo Building GUI version...
    %PYTHON_CMD% -m PyInstaller --name OllamaLink-GUI --onefile --windowed --distpath=dist --workpath=build --icon=icon.ico --add-data "config.json;." run_gui.py
    
    if not exist dist\OllamaLink-GUI.exe (
        echo Error: GUI build failed - executable not created
        exit /b 1
    )
    
    echo GUI executable built successfully.
    
    REM Clean build directory but keep dist
    echo Cleaning build directory...
    if exist build rmdir /s /q build
    
    REM Build CLI version
    echo Building CLI version...
    %PYTHON_CMD% -m PyInstaller --name OllamaLink-CLI --onefile --console --distpath=dist --workpath=build --icon=icon.ico --add-data "config.json;." run_cli.py
    
    if not exist dist\OllamaLink-CLI.exe (
        echo Error: CLI build failed - executable not created
        exit /b 1
    )
    
    echo CLI executable built successfully.
    echo Build complete! Both executables are available in the dist\ directory.
) else (
    REM Default to GUI
    echo Building GUI version...
    %PYTHON_CMD% -m PyInstaller --name OllamaLink-GUI --onefile --windowed --distpath=dist --workpath=build --icon=icon.ico --add-data "config.json;." run_gui.py
    
    if not exist dist\OllamaLink-GUI.exe (
        echo Error: GUI build failed - executable not created
        exit /b 1
    )
    
    echo Build complete! The GUI executable is available in the dist\ directory.
)

echo To build CLI version, run: build_win.bat cli
echo To build both versions, run: build_win.bat all 