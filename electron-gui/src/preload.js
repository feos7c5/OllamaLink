const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Server communication
  getServerPort: () => ipcRenderer.invoke('get-server-port'),
  
  // Persistent storage
  getStoreValue: (key, defaultValue) => ipcRenderer.invoke('get-store-value', key, defaultValue),
  setStoreValue: (key, value) => ipcRenderer.invoke('set-store-value', key, value),
  deleteStoreValue: (key) => ipcRenderer.invoke('delete-store-value', key),
  
  // API requests
  makeApiRequest: (options) => ipcRenderer.invoke('make-api-request', options),
  
  // File dialogs
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
  showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
  showMessageBox: (options) => ipcRenderer.invoke('show-message-box', options),
  
  // Tunnel management
  startTunnel: (port) => ipcRenderer.invoke('start-tunnel', port),
  stopTunnel: () => ipcRenderer.invoke('stop-tunnel'),
  getTunnelStatus: () => ipcRenderer.invoke('get-tunnel-status'),
  onTunnelUrlDetected: (callback) => ipcRenderer.on('tunnel-url-detected', callback),
  onTunnelLog: (callback) => ipcRenderer.on('tunnel-log', callback),
  onTunnelClosed: (callback) => ipcRenderer.on('tunnel-closed', callback),
  onTunnelStopped: (callback) => ipcRenderer.on('tunnel-stopped', callback),
  
  // Backend control
  restartBackend: () => ipcRenderer.invoke('restart-backend'),
  
  // Config file management
  loadConfig: () => ipcRenderer.invoke('load-config'),
  saveConfig: (config) => ipcRenderer.invoke('save-config', config),
  resetConfig: () => ipcRenderer.invoke('reset-config'),
  
  // Event listeners
  onBackendStarted: (callback) => ipcRenderer.on('backend-started', callback),
  onPythonLog: (callback) => ipcRenderer.on('python-log', callback),
  onPythonProcessClosed: (callback) => ipcRenderer.on('python-process-closed', callback),
  
  // Remove listeners
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});
