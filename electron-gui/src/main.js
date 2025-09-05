const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const axios = require('axios');
const Store = require('electron-store');

// Initialize electron-store for persistent settings
const store = new Store();

let mainWindow;
let pythonProcess;
let tunnelProcess;
let serverPort = 8000;
let tunnelUrl = null;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, '../assets/icon.png'),
    titleBarStyle: 'default',
    show: false // Don't show until ready
  });

  // Load the app
  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
    if (pythonProcess) {
      pythonProcess.kill();
    }
  });

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// Find available port
async function findAvailablePort(startPort = 8000) {
  const net = require('net');
  
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.listen(startPort, () => {
      const port = server.address().port;
      server.close(() => resolve(port));
    });
    
    server.on('error', () => {
      resolve(findAvailablePort(startPort + 1));
    });
  });
}

// Find Python executable
function findPythonExecutable() {
  const { spawnSync } = require('child_process');
  const possiblePaths = [
    'python3',
    'python',
    '/usr/bin/python3',
    '/usr/bin/python',
    '/usr/local/bin/python3',
    '/usr/local/bin/python',
    '/opt/homebrew/bin/python3',
    '/System/Library/Frameworks/Python.framework/Versions/3.11/bin/python3',
    '/Library/Frameworks/Python.framework/Versions/3.11/bin/python3'
  ];
  
  for (const pythonCmd of possiblePaths) {
    try {
      console.log(`Testing Python command: ${pythonCmd}`);
      const result = spawnSync(pythonCmd, ['--version'], { 
        stdio: 'pipe',
        timeout: 5000
      });
      
      if (result.status === 0) {
        const version = result.stdout.toString().trim() || result.stderr.toString().trim();
        console.log(`Found working Python: ${pythonCmd} (${version})`);
        return pythonCmd;
      }
    } catch (error) {
      console.log(`Failed to test ${pythonCmd}: ${error.message}`);
      continue;
    }
  }
  
  console.error('No working Python installation found!');
  throw new Error('Python not found. Please ensure Python 3.8+ is installed and accessible.');
}

// Start Python backend
async function startPythonBackend() {
  try {
    serverPort = await findAvailablePort(8000);
    
    // Find Python executable
    const pythonCmd = findPythonExecutable();
    console.log(`Using Python command: ${pythonCmd}`);
    
    // Path to the Python backend
    const pythonPath = path.join(__dirname, '../../run_api.py');
    
    pythonProcess = spawn(pythonCmd, [pythonPath, '--port', serverPort.toString()], {
      cwd: path.join(__dirname, '../..'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
      if (mainWindow) {
        mainWindow.webContents.send('python-log', { 
          type: 'stdout', 
          message: data.toString() 
        });
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
      if (mainWindow) {
        mainWindow.webContents.send('python-log', { 
          type: 'stderr', 
          message: data.toString() 
        });
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      if (mainWindow) {
        mainWindow.webContents.send('python-process-closed', code);
      }
    });

    // Wait a moment for server to start
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    return serverPort;
  } catch (error) {
    console.error('Failed to start Python backend:', error);
    throw error;
  }
}

// App event handlers
app.whenReady().then(async () => {
  createWindow();
  
  try {
    const port = await startPythonBackend();
    mainWindow.webContents.send('backend-started', { port });
  } catch (error) {
    dialog.showErrorBox('Backend Error', `Failed to start Python backend: ${error.message}`);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (tunnelProcess) {
    tunnelProcess.kill();
  }
});

// Tunnel Management Functions - Using Backend API
async function startTunnel(port) {
  try {
    console.log('Starting tunnel via backend API...');
    
    const response = await axios.post(`http://localhost:${serverPort}/api/tunnel/start`, {
      port: port
    });
    
    if (response.data.success) {
      tunnelUrl = response.data.tunnel_url;
      console.log('Tunnel started successfully:', tunnelUrl);
      console.log('Cursor AI URL:', response.data.cursor_url);
      
      if (mainWindow) {
        mainWindow.webContents.send('tunnel-url-detected', { 
          url: response.data.tunnel_url,
          cursor_url: response.data.cursor_url 
        });
      }
      
      return { 
        url: response.data.tunnel_url, 
        cursor_url: response.data.cursor_url 
      };
    } else {
      throw new Error('Failed to start tunnel via backend API');
    }
    
  } catch (error) {
    console.error('Failed to start tunnel:', error);
    
    if (mainWindow) {
      mainWindow.webContents.send('tunnel-log', { 
        message: `Error: ${error.message}`, 
        type: 'error' 
      });
    }
    
    throw error;
  }
}

async function stopTunnel() {
  try {
    console.log('Stopping tunnel via backend API...');
    
    const response = await axios.post(`http://localhost:${serverPort}/api/tunnel/stop`);
    
    if (response.data.success) {
      tunnelProcess = null;
      tunnelUrl = null;
      console.log('Tunnel stopped successfully');
      
      if (mainWindow) {
        mainWindow.webContents.send('tunnel-stopped');
      }
    } else {
      throw new Error('Failed to stop tunnel via backend API');
    }
    
  } catch (error) {
    console.error('Failed to stop tunnel:', error);
    
    if (mainWindow) {
      mainWindow.webContents.send('tunnel-log', { 
        message: `Error stopping tunnel: ${error.message}`, 
        type: 'error' 
      });
    }
    
    throw error;
  }
}

// IPC handlers
ipcMain.handle('get-server-port', () => {
  return serverPort;
});

ipcMain.handle('get-store-value', (event, key, defaultValue) => {
  return store.get(key, defaultValue);
});

ipcMain.handle('set-store-value', (event, key, value) => {
  store.set(key, value);
  return true;
});

ipcMain.handle('delete-store-value', (event, key) => {
  store.delete(key);
  return true;
});

ipcMain.handle('make-api-request', async (event, options) => {
  try {
    const response = await axios(options);
    return {
      success: true,
      data: response.data,
      status: response.status,
      headers: response.headers
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      status: error.response?.status,
      data: error.response?.data
    };
  }
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('restart-backend', async () => {
  try {
    if (pythonProcess) {
      pythonProcess.kill();
    }
    
    const port = await startPythonBackend();
    return { success: true, port };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// Config file management
const getConfigPath = () => path.join(__dirname, '../../config.json');

ipcMain.handle('load-config', async () => {
  try {
    const configPath = getConfigPath();
    const configData = fs.readFileSync(configPath, 'utf8');
    return { success: true, config: JSON.parse(configData) };
  } catch (error) {
    console.error('Error loading config:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('save-config', async (event, config) => {
  try {
    const configPath = getConfigPath();
    // Backup existing config
    const backupPath = configPath + '.backup';
    if (fs.existsSync(configPath)) {
      fs.copyFileSync(configPath, backupPath);
    }
    
    fs.writeFileSync(configPath, JSON.stringify(config, null, 4));
    return { success: true };
  } catch (error) {
    console.error('Error saving config:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('reset-config', async () => {
  try {
    const configPath = getConfigPath();
    const backupPath = configPath + '.backup';
    
    if (fs.existsSync(backupPath)) {
      fs.copyFileSync(backupPath, configPath);
      return { success: true };
    } else {
      return { success: false, error: 'No backup file found' };
    }
  } catch (error) {
    console.error('Error resetting config:', error);
    return { success: false, error: error.message };
  }
});

// Tunnel IPC handlers
ipcMain.handle('start-tunnel', async (event, port = serverPort) => {
  try {
    const result = await startTunnel(port);
    tunnelUrl = result.url;
    return { success: true, url: result.url };
  } catch (error) {
    console.error('Failed to start tunnel:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('stop-tunnel', () => {
  try {
    stopTunnel();
    return { success: true };
  } catch (error) {
    console.error('Failed to stop tunnel:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-tunnel-status', () => {
  return {
    running: !!tunnelProcess,
    url: tunnelUrl
  };
});
