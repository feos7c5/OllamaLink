#!/bin/bash
# Build script for macOS using py2app with PyQt6 fixes

# Find the script directory (where build_mac.sh is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Detect Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Setup environment
export PYTHONPATH="$($PYTHON_CMD -c 'import sys; print(":".join(sys.path))')"
export QT_PLUGIN_PATH="$($PYTHON_CMD -c 'import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins"))')"
export DYLD_FRAMEWORK_PATH="$($PYTHON_CMD -c 'import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "lib"))')"

# Create dist directory in root if it doesn't exist
mkdir -p dist

# Install required dependencies
echo "Installing required dependencies..."
$PYTHON_CMD -m pip install py2app
$PYTHON_CMD -m pip install -r requirements.txt

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist build2app_build

# Function to fix Qt dependencies in app bundle
fix_qt_bundle() {
    local APP_PATH="$1"
    
    # Only fix Qt dependencies for GUI app
    if [[ "$APP_PATH" == *"CLI"* ]]; then
        echo "Skipping Qt fixes for CLI version"
        return
    fi
    
    echo "Fixing Qt dependencies for $APP_PATH..."
    
    # Create frameworks directory
    mkdir -p "$APP_PATH/Contents/Frameworks"

    # Copy Qt plugins
    QT_PLUGINS_PATH="$($PYTHON_CMD -c 'import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins"))')"
    mkdir -p "$APP_PATH/Contents/PlugIns"
    cp -R "$QT_PLUGINS_PATH/platforms" "$APP_PATH/Contents/PlugIns/"
    cp -R "$QT_PLUGINS_PATH/styles" "$APP_PATH/Contents/PlugIns/"

    # Copy Qt frameworks
    QT_LIBS_PATH="$($PYTHON_CMD -c 'import os, PyQt6; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "lib"))')"
    for framework in QtCore QtGui QtWidgets; do
        cp -R "$QT_LIBS_PATH/${framework}.framework" "$APP_PATH/Contents/Frameworks/"
    done

    # Update runtime paths in the binaries
    $PYTHON_CMD -c "
    import os
    import subprocess
    import glob

    app_path = '$APP_PATH'
    frameworks = glob.glob(f'{app_path}/Contents/Frameworks/*.framework/Versions/A/*')
    plugins = glob.glob(f'{app_path}/Contents/PlugIns/*/*.dylib')
    pyqt_libs = glob.glob(f'{app_path}/Contents/Resources/lib/python*/PyQt6/*.so')

    for binary in frameworks + plugins + pyqt_libs:
        if os.path.islink(binary) or not os.path.isfile(binary):
            continue
        
        print(f'Fixing {binary}')
        subprocess.run(['install_name_tool', '-id', f'@executable_path/../Frameworks/{os.path.basename(binary)}', binary])
        
        otool = subprocess.check_output(['otool', '-L', binary]).decode()
        for line in otool.splitlines():
            if 'Qt6' in line and '@rpath' in line:
                old_path = line.split()[0]
                framework = old_path.split('/')[-3]
                new_path = f'@executable_path/../Frameworks/{framework}/Versions/A/{framework}'
                subprocess.run(['install_name_tool', '-change', old_path, new_path, binary])
    "
    
    # Add Qt plugins path to Info.plist
    /usr/libexec/PlistBuddy -c "Add :LSEnvironment dict" "$APP_PATH/Contents/Info.plist" 2>/dev/null || true
    /usr/libexec/PlistBuddy -c "Add :LSEnvironment:QT_PLUGIN_PATH string @executable_path/../PlugIns" "$APP_PATH/Contents/Info.plist"
}

# Build options
if [ "$1" == "cli" ]; then
    echo "Building CLI version..."
    $PYTHON_CMD setup.py py2app --cli
    
    # Check if build was successful
    if [ ! -d "dist/OllamaLink-CLI.app" ]; then
        echo "Error: Build failed - app bundle not created"
        exit 1
    fi
    
    # Cleanup temporary folders
    echo "Cleaning up temporary build folders..."
    rm -rf build2app_build
    
    echo "Build complete! The CLI application is available in the dist/ directory."
    echo "Run with: open dist/OllamaLink-CLI.app"
elif [ "$1" == "all" ]; then
    echo "Building both GUI and CLI versions..."
    
    # Build GUI version first (requires PyQt6)
    echo "Building GUI version..."
    $PYTHON_CMD setup.py py2app
    
    # Check if GUI build was successful
    if [ ! -d "dist/OllamaLink.app" ]; then
        echo "Error: GUI build failed - app bundle not created"
        exit 1
    fi
    
    # Fix Qt dependencies for GUI version
    fix_qt_bundle "dist/OllamaLink.app"
    
    # Cleanup temporary folders after GUI build
    echo "Cleaning up temporary build folders..."
    rm -rf build2app_build
    
    echo "GUI application built successfully."
    
    # Clean build directory but keep dist
    echo "Cleaning build directory..."
    rm -rf build
    
    # Build CLI version
    echo "Building CLI version..."
    $PYTHON_CMD setup.py py2app --cli
    
    # Check if CLI build was successful
    if [ ! -d "dist/OllamaLink-CLI.app" ]; then
        echo "Error: CLI build failed - app bundle not created"
        exit 1
    fi
    
    # Cleanup temporary folders after CLI build
    echo "Cleaning up temporary build folders..."
    rm -rf build2app_build
    
    echo "CLI application built successfully."
    echo "Build complete! Both applications are available in the dist/ directory."
    echo "Run GUI with: open dist/OllamaLink.app"
    echo "Run CLI with: open dist/OllamaLink-CLI.app"
else
    # Default to GUI
    echo "Building GUI version..."
    $PYTHON_CMD setup.py py2app
    
    # Check if build was successful
    if [ ! -d "dist/OllamaLink.app" ]; then
        echo "Error: Build failed - app bundle not created"
        exit 1
    fi
    
    # Fix Qt dependencies for GUI version
    fix_qt_bundle "dist/OllamaLink.app"
    
    # Cleanup temporary folders
    echo "Cleaning up temporary build folders..."
    rm -rf build2app_build
    
    echo "Build complete! The GUI application is available in the dist/ directory."
    echo "Run with: open dist/OllamaLink.app"
fi

echo "To build CLI version, run: ./build_mac.sh cli"
echo "To build both versions, run: ./build_mac.sh all"