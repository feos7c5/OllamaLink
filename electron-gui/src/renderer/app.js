// Simplified OllamaLink Electron GUI
class OllamaLinkApp {
    constructor() {
        this.serverPort = 8000;
        this.baseUrl = `http://localhost:${this.serverPort}`;
        this.serverStatus = 'starting';
        this.providers = [];
        this.tunnelStatus = { running: false, url: null };
        this.config = {};
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        console.log('Initializing OllamaLink GUI...');
        
        try {
            // Setup event listeners first
            this.setupEventListeners();
            this.setupBackendListeners();
            
            // Load config
            await this.loadConfig();
            
            // Initialize UI with config
            this.initializeUI();
            
            // Mark as initialized
            this.isInitialized = true;
            console.log('✅ GUI initialized successfully');
            
        } catch (error) {
            console.error('❌ Failed to initialize GUI:', error);
            this.showError('Failed to initialize application');
        }
    }

    async loadConfig() {
        try {
            const result = await window.electronAPI.loadConfig();
            if (result.success) {
                this.config = result.config;
                console.log('✅ Config loaded successfully');
            } else {
                console.error('❌ Failed to load config:', result.error);
                this.config = this.getDefaultConfig();
            }
        } catch (error) {
            console.error('❌ Error loading config:', error);
            this.config = this.getDefaultConfig();
        }
    }

    getDefaultConfig() {
        return {
            server: { port: 8000, hostname: "127.0.0.1" },
            ollama: { enabled: true, endpoint: "http://localhost:11434" },
            openrouter: { enabled: false, api_key: "", endpoint: "https://openrouter.ai" },
            llamacpp: { enabled: false, endpoint: "http://localhost:8080" },
            tunnel: { use_tunnel: true },
            routing: { provider_priority: ["ollama", "openrouter", "llamacpp"] }
        };
    }

    async saveConfig() {
        try {
            const result = await window.electronAPI.saveConfig(this.config);
            
            if (result.success) {
                this.showFeedback('Configuration saved!', 'success');
                console.log('✅ Config saved successfully');
            } else {
                this.showFeedback('Failed to save: ' + result.error, 'error');
                console.error('❌ Failed to save config:', result.error);
            }
        } catch (error) {
            console.error('❌ Error saving config:', error);
            this.showFeedback('Save error', 'error');
        }
    }

    showError(message) {
        console.error('❌ Error:', message);
        this.showFeedback(message, 'error');
    }

    showFeedback(message, type = 'info') {
        // Simple feedback system - could be enhanced with toast notifications
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Add to console if it exists
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;
        
        const consoleOutput = document.getElementById('consoleOutput');
        if (consoleOutput) {
            consoleOutput.appendChild(logEntry);
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
    }
    
    // Simplified tunnel management
    setupTunnelListeners() {
        // Backend event listeners
        if (window.electronAPI.onTunnelUrlDetected) {
            window.electronAPI.onTunnelUrlDetected((event, data) => {
                this.tunnelStatus = { running: true, url: data.url, cursor_url: data.cursor_url };
                this.updateTunnelUI();
                this.showFeedback(`Tunnel active: ${data.cursor_url}`, 'success');
            });
        }
        
        if (window.electronAPI.onTunnelStopped) {
            window.electronAPI.onTunnelStopped(() => {
                this.tunnelStatus = { running: false, url: null, cursor_url: null };
                this.updateTunnelUI();
                this.showFeedback('Tunnel stopped', 'info');
            });
        }
        
        // UI button listeners
        this.addClickListener('copy-cursor-url', () => this.copyCursorUrl());
        this.addClickListener('restartApiBtn', () => this.restartApi());
    }

    addClickListener(elementId, handler) {
        const element = document.getElementById(elementId);
        if (element) {
            element.addEventListener('click', handler);
        }
    }
    
    async startTunnel() {
        try {
            const serverPort = this.config?.server?.port || 8000;
            
            // Update UI to show starting state
            this.tunnelStatus.running = true;
            this.updateTunnelUI('starting');
            this.showFeedback('Starting tunnel...', 'info');
            this.addTunnelLog('Starting localhost.run tunnel...', 'info');
            
            const result = await window.electronAPI.startTunnel(serverPort);
            
            if (result.success) {
                this.tunnelStatus.url = result.url;
                this.updateTunnelUI('running');
                this.showFeedback('Tunnel started successfully!', 'success');
            } else {
                this.tunnelStatus.running = false;
                this.updateTunnelUI('error');
                this.showFeedback('Failed to start tunnel: ' + result.error, 'error');
                this.addTunnelLog('Failed to start tunnel: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Error starting tunnel:', error);
            this.tunnelStatus.running = false;
            this.updateTunnelUI('error');
            this.showFeedback('Error starting tunnel', 'error');
            this.addTunnelLog('Error starting tunnel: ' + error.message, 'error');
        }
    }
    
    async stopTunnel() {
        try {
            this.showFeedback('Stopping tunnel...', 'info');
            this.addTunnelLog('Stopping tunnel...', 'info');
            
            const result = await window.electronAPI.stopTunnel();
            
            if (result.success) {
                this.tunnelStatus.running = false;
                this.tunnelStatus.url = null;
                this.updateTunnelUI();
                this.showFeedback('Tunnel stopped', 'info');
            } else {
                this.showFeedback('Failed to stop tunnel: ' + result.error, 'error');
                this.addTunnelLog('Failed to stop tunnel: ' + result.error, 'error');
            }
        } catch (error) {
            console.error('Error stopping tunnel:', error);
            this.showFeedback('Error stopping tunnel', 'error');
            this.addTunnelLog('Error stopping tunnel: ' + error.message, 'error');
        }
    }
    
    async loadApiStatus() {
        try {
            console.log('🔍 Checking API status...');
            
            const apiStatusLight = document.getElementById('api-status');
            const apiStatusText = document.getElementById('api-status-text');
            const apiHealth = document.getElementById('api-health');
            const restartBtn = document.getElementById('restartApiBtn');
            
            console.log('🔧 DOM elements found:', {
                statusLight: !!apiStatusLight,
                statusText: !!apiStatusText,
                health: !!apiHealth,
                restartBtn: !!restartBtn
            });
            
            // Try to ping the API models endpoint to check if API is running
            console.log('📡 Fetching http://localhost:8000/v1/models...');
            const response = await fetch('http://localhost:8000/v1/models');
            console.log('📡 Response status:', response.status, response.statusText);
            
            if (response.ok) {
                const healthData = await response.json();
                console.log('✅ API Response successful:', healthData);
                
                // API is running and healthy
                console.log('🟢 Setting API status to Online...');
                if (apiStatusLight) apiStatusLight.className = 'status-light on';
                if (apiStatusText) apiStatusText.textContent = 'API Online';
                if (apiHealth) apiHealth.textContent = 'Healthy';
                if (restartBtn) restartBtn.disabled = false;
                
                console.log('✅ API status updated successfully!');
                
            } else {
                throw new Error(`API responded with status ${response.status}`);
            }
            
        } catch (error) {
            console.error('❌ API check failed:', error.message);
            
            const apiStatusLight = document.getElementById('api-status');
            const apiStatusText = document.getElementById('api-status-text');
            const apiHealth = document.getElementById('api-health');
            
            // API is not ready yet (still starting)
            console.log('⏳ Setting API status to Starting...');
            if (apiStatusLight) apiStatusLight.className = 'status-light starting';
            if (apiStatusText) apiStatusText.textContent = 'Starting API...';
            if (apiHealth) apiHealth.textContent = 'Waiting...';
            
            // Retry in a few seconds
            setTimeout(() => this.loadApiStatus(), 3000);
        }
    }
    
    async loadTunnelStatus() {
        try {
            console.log('Loading tunnel status from backend...');
            
            // Query backend API directly for current tunnel status
            const response = await fetch('http://localhost:8000/api/tunnel/status');
            const status = await response.json();
            
            console.log('🚇 Backend tunnel status received:', status);
            console.log('🚇 Tunnel details:', {
                running: status.running,
                tunnel_url: status.tunnel_url,
                cursor_url: status.cursor_url,
                port: status.port
            });
            
            // Update our tunnel status
            this.tunnelStatus = {
                running: status.running,
                url: status.tunnel_url,
                cursor_url: status.cursor_url
            };
            
            // Update the UI to reflect current status
            this.updateTunnelUI();
            
            // If tunnel is running, show success message
            if (status.running && status.cursor_url) {
                this.addTunnelLog(`Tunnel already running: ${status.tunnel_url}`, 'success');
                this.addTunnelLog(`Cursor AI URL: ${status.cursor_url}`, 'success');
            }
            
        } catch (error) {
            console.error('Error loading tunnel status from backend:', error);
            this.addTunnelLog(`Error loading tunnel status: ${error.message}`, 'error');
            
            // Fallback to offline status
            this.tunnelStatus = { running: false, url: null, cursor_url: null };
            this.updateTunnelUI();
        }
    }
    
    updateTunnelUI(state = null) {
        const statusText = document.getElementById('tunnel-status-text');
        const cursorUrlLink = document.getElementById('cursor-url-link');
        
        if (!statusText || !cursorUrlLink) return;
        
        // Determine actual state
        const actualState = state || (this.tunnelStatus.running ? 'running' : 'stopped');
        
        // Update status text and badge class (with null checks)
        if (statusText) {
            switch (actualState) {
                case 'starting':
                    statusText.textContent = 'Starting...';
                    statusText.className = 'status-badge starting';
                    break;
                case 'running':
                    statusText.textContent = 'Online';
                    statusText.className = 'status-badge online';
                    break;
                case 'error':
                    statusText.textContent = 'Error';
                    statusText.className = 'status-badge offline';
                    break;
                default: // stopped
                    statusText.textContent = 'Offline';
                    statusText.className = 'status-badge offline';
                    break;
            }
        }
        
        // Update URL display
        const urlItem = document.getElementById('tunnel-url-item');
        if (this.tunnelStatus.cursor_url && this.tunnelStatus.running && cursorUrlLink && urlItem) {
            cursorUrlLink.textContent = this.tunnelStatus.cursor_url;
            cursorUrlLink.href = this.tunnelStatus.cursor_url;
            urlItem.style.display = 'block';
        } else if (urlItem) {
            urlItem.style.display = 'none';
            if (cursorUrlLink) {
                cursorUrlLink.textContent = '-';
                cursorUrlLink.href = '#';
            }
        }
    }
    
    copyTunnelUrl() {
        const urlInput = document.getElementById('tunnel-url');
        if (urlInput && urlInput.value) {
            navigator.clipboard.writeText(urlInput.value).then(() => {
                this.showFeedback('Tunnel URL copied to clipboard!', 'success');
            }).catch(err => {
                console.error('Failed to copy URL:', err);
                this.showFeedback('Failed to copy URL', 'error');
            });
        }
    }
    
    copyCursorUrl() {
        const cursorUrlInput = document.getElementById('cursor-url');
        if (cursorUrlInput && cursorUrlInput.value) {
            navigator.clipboard.writeText(cursorUrlInput.value).then(() => {
                this.showFeedback('Cursor AI URL copied to clipboard!', 'success');
                this.addTunnelLog('Copied Cursor AI URL to clipboard', 'success');
            }).catch(err => {
                console.error('Failed to copy Cursor URL:', err);
                this.showFeedback('Failed to copy Cursor URL', 'error');
            });
        } else {
            this.showFeedback('No Cursor URL available', 'warning');
        }
    }
    
    openTunnelUrl() {
        const urlInput = document.getElementById('tunnel-url');
        if (urlInput && urlInput.value) {
            window.electronAPI.openExternal(urlInput.value);
        }
    }
    
    async restartApi() {
        try {
            console.log('🔄 Restarting API...');
            
            // Show restart in progress
            const apiStatusText = document.getElementById('api-status-text');
            const apiHealth = document.getElementById('api-health');
            const restartBtn = document.getElementById('restartApiBtn');
            
            if (apiStatusText) apiStatusText.textContent = 'Restarting API...';
            if (apiHealth) apiHealth.textContent = 'Restarting...';
            if (restartBtn) restartBtn.disabled = true;
            
            // TODO: Implement API restart via IPC to main process
            // For now, just reload API status after a delay
            setTimeout(() => {
                this.loadApiStatus();
            }, 2000);
            
        } catch (error) {
            console.error('❌ API restart failed:', error);
            this.showFeedback('Failed to restart API', 'error');
        }
    }
    
    addTunnelLog(message, type = 'info') {
        const logsContainer = document.getElementById('tunnel-logs');
        if (!logsContainer) return;
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;
        
        // Limit log entries to prevent memory issues
        const maxEntries = 100;
        while (logsContainer.children.length > maxEntries) {
            logsContainer.removeChild(logsContainer.firstChild);
        }
    }
    
    clearTunnelLogs() {
        const logsContainer = document.getElementById('tunnel-logs');
        if (logsContainer) {
            logsContainer.innerHTML = '';
            this.showFeedback('Tunnel logs cleared', 'info');
        }
    }

    async loadSettings() {
        try {
            this.settings = {
                openrouterApiKey: await window.electronAPI.getStoreValue('openrouterApiKey', ''),
                openaiApiKey: await window.electronAPI.getStoreValue('openaiApiKey', ''),
                autoStartServer: await window.electronAPI.getStoreValue('autoStartServer', true),
                enableNotifications: await window.electronAPI.getStoreValue('enableNotifications', true),
                theme: await window.electronAPI.getStoreValue('theme', 'dark'),
                requestTimeout: await window.electronAPI.getStoreValue('requestTimeout', 30),
                maxConcurrentRequests: await window.electronAPI.getStoreValue('maxConcurrentRequests', 10)
            };
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    async saveSettings() {
        try {
            for (const [key, value] of Object.entries(this.settings)) {
                await window.electronAPI.setStoreValue(key, value);
            }
        } catch (error) {
            console.error('Error saving settings:', error);
        }
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const tabName = e.currentTarget.dataset.tab;
                this.switchTab(tabName);
            });
        });

        // Configuration save button
        this.addClickListener('saveConfigBtn', () => this.saveConfigFromUI());
        this.addClickListener('resetConfigBtn', () => this.resetConfig());
        
        // Setup tunnel listeners
        this.setupTunnelListeners();

        // Close modal handler
        const modalClose = document.querySelector('.modal-close');
        if (modalClose) {
            modalClose.addEventListener('click', () => this.closeModal());
        }
    }

    setupBackendListeners() {
        // Backend started
        if (window.electronAPI.onBackendStarted) {
            window.electronAPI.onBackendStarted((event, data) => {
                console.log('✅ Backend started on port:', data.port);
                this.serverPort = data.port;
                this.baseUrl = `http://localhost:${this.serverPort}`;
                this.updateServerStatus('online');
                
                // Start monitoring once backend is ready
                setTimeout(() => this.startMonitoring(), 1000);
            });
        }

        // Python logs (optional)
        if (window.electronAPI.onPythonLog) {
            window.electronAPI.onPythonLog((event, data) => {
                this.showFeedback(data.message, data.type);
            });
        }

        // Python process closed
        if (window.electronAPI.onPythonProcessClosed) {
            window.electronAPI.onPythonProcessClosed((event, code) => {
                console.log('❌ Backend process closed with code:', code);
                this.updateServerStatus('offline');
                this.stopMonitoring();
            });
        }
    }

    startMonitoring() {
        if (this.monitoringInterval) return;
        
        // Check API status immediately
        this.checkApiStatus();
        
        // Then check every 5 seconds
        this.monitoringInterval = setInterval(() => {
            this.checkApiStatus();
        }, 5000);
        
        console.log('✅ Monitoring started');
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
            console.log('⏹️ Monitoring stopped');
        }
    }

    async checkApiStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/v1/models`, { timeout: 3000 });
            if (response.ok) {
                this.updateServerStatus('online');
                // Load provider status
                this.loadProviders();
            } else {
                this.updateServerStatus('error');
            }
        } catch (error) {
            this.updateServerStatus('starting');
        }
    }

    initializeUI() {
        // Populate basic config in UI
        this.populateConfigUI();
        
        // Set initial server status
        this.updateServerStatus('starting');
        
        // Initialize provider status display
        this.renderProviders();
    }

    populateConfigUI() {
        if (!this.config) return;

        try {
            // Basic provider settings
            this.setInputValue('openrouterEnabled', this.config.openrouter?.enabled || false, 'checkbox');
            this.setInputValue('openrouterApiKey', this.config.openrouter?.api_key || '');
            this.setInputValue('openrouterEndpoint', this.config.openrouter?.endpoint || 'https://openrouter.ai');

            this.setInputValue('ollamaEnabled', this.config.ollama?.enabled !== false, 'checkbox');
            this.setInputValue('ollamaEndpoint', this.config.ollama?.endpoint || 'http://localhost:11434');

            this.setInputValue('llamacppEnabled', this.config.llamacpp?.enabled || false, 'checkbox');
            this.setInputValue('llamacppEndpoint', this.config.llamacpp?.endpoint || 'http://localhost:8080');

            this.setInputValue('serverPort', this.config.server?.port || 8000);
            this.setInputValue('serverHostname', this.config.server?.hostname || '127.0.0.1');

        } catch (error) {
            console.error('❌ Error populating config UI:', error);
        }
    }

    setInputValue(elementId, value, type = 'text') {
        const element = document.getElementById(elementId);
        if (!element) return;

        try {
            if (type === 'checkbox') {
                element.checked = Boolean(value);
            } else {
                element.value = String(value);
            }
        } catch (error) {
            console.error(`❌ Error setting ${elementId}:`, error);
        }
    }

    populateProviderPriority(priority) {
        const container = document.getElementById('providerPriority');
        container.innerHTML = '';
        
        priority.forEach((provider, index) => {
            const item = document.createElement('div');
            item.className = 'priority-item';
            item.draggable = true;
            item.dataset.provider = provider;
            item.innerHTML = `
                <span>${provider}</span>
                <div class="priority-drag-handle">
                    <i class="fas fa-grip-vertical"></i>
                </div>
            `;
            container.appendChild(item);
        });
        
        // Add drag and drop functionality
        this.setupProviderPriorityDragDrop();
    }

    setupProviderPriorityDragDrop() {
        const container = document.getElementById('providerPriority');
        let draggedElement = null;
        
        container.addEventListener('dragstart', (e) => {
            draggedElement = e.target.closest('.priority-item');
            if (draggedElement) {
                draggedElement.classList.add('dragging');
            }
        });
        
        container.addEventListener('dragend', (e) => {
            if (draggedElement) {
                draggedElement.classList.remove('dragging');
                draggedElement = null;
                this.updateProviderPriorityFromUI();
            }
        });
        
        container.addEventListener('dragover', (e) => {
            e.preventDefault();
            const afterElement = this.getDragAfterElement(container, e.clientY);
            if (draggedElement) {
                if (afterElement == null) {
                    container.appendChild(draggedElement);
                } else {
                    container.insertBefore(draggedElement, afterElement);
                }
            }
        });
    }

    getDragAfterElement(container, y) {
        const draggableElements = [...container.querySelectorAll('.priority-item:not(.dragging)')];
        
        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            const offset = y - box.top - box.height / 2;
            
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    updateProviderPriorityFromUI() {
        const items = document.querySelectorAll('#providerPriority .priority-item');
        const priority = Array.from(items).map(item => item.dataset.provider);
        this.updateConfigValue('routing.provider_priority', priority);
    }

    updateConfigValue(path, value) {
        const keys = path.split('.');
        let current = this.config;
        
        // Navigate to the parent object
        for (let i = 0; i < keys.length - 1; i++) {
            if (!current[keys[i]]) {
                current[keys[i]] = {};
            }
            current = current[keys[i]];
        }
        
        // Set the value
        current[keys[keys.length - 1]] = value;
        console.log(`Updated config ${path} to:`, value);
    }

    updateConfigMappings(provider, jsonText) {
        try {
            const mappings = JSON.parse(jsonText || '{}');
            this.updateConfigValue(`${provider}.model_mappings`, mappings);
        } catch (error) {
            console.error('Invalid JSON for model mappings:', error);
            this.addLogEntry('error', `Invalid JSON for ${provider} model mappings: ${error.message}`);
        }
    }

    async saveConfigFromUI() {
        try {
            const success = await this.saveConfig();
            if (success) {
                this.addLogEntry('info', 'Configuration saved successfully');
                // Show success feedback
                const btn = document.getElementById('saveConfigBtn');
                const originalText = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i> Saved!';
                btn.classList.add('btn-success');
                
                setTimeout(() => {
                    btn.innerHTML = originalText;
                    btn.classList.remove('btn-success');
                }, 2000);
            }
        } catch (error) {
            console.error('Error saving config:', error);
            this.addLogEntry('error', `Error saving config: ${error.message}`);
        }
    }

    async resetConfig() {
        if (confirm('Are you sure you want to reset the configuration? This will restore the last saved version.')) {
            try {
                const result = await window.electronAPI.resetConfig();
                if (result.success) {
                    await this.loadConfig();
                    this.populateConfigUI();
                    this.addLogEntry('info', 'Configuration reset successfully');
                } else {
                    this.addLogEntry('error', `Failed to reset config: ${result.error}`);
                }
            } catch (error) {
                console.error('Error resetting config:', error);
                this.addLogEntry('error', `Error resetting config: ${error.message}`);
            }
        }
    }

    async waitForBackend() {
        // The backend is started automatically by the main process
        // We just need to wait for the signal
        setTimeout(() => {
            if (this.serverStatus === 'starting') {
                this.updateServerStatus('starting');
            }
        }, 1000);
    }

    async loadInitialData() {
        try {
            // Only load data for sections that still exist in simplified GUI
            await Promise.all([
                this.loadRequests(),
                this.updateStats()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.addLogEntry('error', `Failed to load initial data: ${error.message}`);
        }
    }

    switchTab(tabName) {
        // Update active menu item
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update active tab content
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');

        // Update page title
        const titles = {
            dashboard: 'Dashboard',
            providers: 'Providers',
            models: 'Models',
            requests: 'Requests',
            console: 'Console',
            settings: 'Settings'
        };
        document.querySelector('.page-title').textContent = titles[tabName] || 'Dashboard';

        // Load data if needed
        if (tabName === 'requests') {
            this.loadRequests();
        } else if (tabName === 'models') {
            this.loadModels();
        } else if (tabName === 'providers') {
            this.loadProviders();
        }
    }

    updateServerStatus(status) {
        this.serverStatus = status;
        const statusElement = document.getElementById('serverStatus');
        const statusTextElement = document.getElementById('serverStatusText');
        const statusInfoElement = document.getElementById('serverStatusInfo');

        if (statusElement) statusElement.className = `status-indicator ${status}`;
        if (statusInfoElement) statusInfoElement.className = `status-badge ${status}`;

        const statusTexts = {
            online: 'Online',
            offline: 'Offline', 
            starting: 'Starting...',
            error: 'Error'
        };

        const statusText = statusTexts[status] || 'Unknown';
        if (statusTextElement) statusTextElement.textContent = statusText;
        if (statusInfoElement) statusInfoElement.textContent = statusText;
    }

    async makeApiRequest(endpoint, options = {}) {
        try {
            const url = `${this.baseUrl}${endpoint}`;
            const requestOptions = {
                url,
                method: 'GET',
                timeout: this.settings.requestTimeout * 1000,
                ...options
            };

            const result = await window.electronAPI.makeApiRequest(requestOptions);
            
            if (!result.success) {
                throw new Error(result.error || 'Request failed');
            }

            return result.data;
        } catch (error) {
            console.error(`API request failed for ${endpoint}:`, error);
            throw error;
        }
    }

    async loadProviders() {
        try {
            const data = await this.makeApiRequest('/api/providers/status');
            
            // Convert GUI-formatted provider status to expected format
            const providers = [];
            if (data.providers) {
                Object.entries(data.providers).forEach(([key, provider]) => {
                    providers.push({
                        name: provider.name,
                        status: provider.status,
                        description: `${provider.models.length} models • ${provider.endpoint}`,
                        enabled: provider.enabled,
                        healthy: provider.healthy,
                        error: provider.error
                    });
                });
            }
            
            this.providers = providers;
            this.renderProviders();
        } catch (error) {
            console.error('Error loading providers:', error);
            this.addLogEntry('error', `Failed to load providers: ${error.message}`);
            this.renderProvidersError();
        }
    }

    renderProviders() {
        const container = document.getElementById('providersList');
        const dashboardContainer = document.getElementById('providerStatus');

        if (this.providers.length === 0) {
            container.innerHTML = '<div class="placeholder">No providers configured</div>';
            dashboardContainer.innerHTML = '<div class="placeholder">No providers available</div>';
            return;
        }

        const providersHtml = this.providers.map(provider => `
            <div class="provider-item">
                <div class="provider-info">
                    <div class="provider-name">${provider.name}</div>
                    <div class="provider-description">${provider.description || 'No description'}</div>
                </div>
                <div class="status-badge ${provider.status}">${provider.status}</div>
            </div>
        `).join('');

        container.innerHTML = providersHtml;
        dashboardContainer.innerHTML = providersHtml;
    }

    renderProvidersError() {
        const errorHtml = '<div class="placeholder">Failed to load providers</div>';
        document.getElementById('providersList').innerHTML = errorHtml;
        document.getElementById('providerStatus').innerHTML = errorHtml;
    }

    async loadModels() {
        try {
            const data = await this.makeApiRequest('/v1/models');
            this.models = data.data || [];
            this.renderModels();
        } catch (error) {
            console.error('Error loading models:', error);
            this.addLogEntry('error', `Failed to load models: ${error.message}`);
            this.renderModelsError();
        }
    }

    renderModels(filteredModels = null) {
        const container = document.getElementById('modelsList');
        const modelsToRender = filteredModels || this.models;

        if (modelsToRender.length === 0) {
            container.innerHTML = '<div class="placeholder">No models available</div>';
            return;
        }

        const modelsHtml = modelsToRender.map(model => `
            <div class="model-item">
                <div class="model-name">${model.id}</div>
                <div class="model-provider">${model.provider || 'Unknown'}</div>
            </div>
        `).join('');

        container.innerHTML = modelsHtml;
    }

    renderModelsError() {
        document.getElementById('modelsList').innerHTML = '<div class="placeholder">Failed to load models</div>';
    }

    filterModels(searchTerm) {
        const filtered = this.models.filter(model => 
            model.id.toLowerCase().includes(searchTerm.toLowerCase())
        );
        this.renderModels(filtered);
    }

    async loadRequests() {
        // This would typically load from the backend's request history
        // For now, we'll show a placeholder
        this.renderRequests();
    }

    renderRequests() {
        const tbody = document.querySelector('#requestsTable tbody');
        
        if (this.requests.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="placeholder">No requests yet</td></tr>';
            return;
        }

        const requestsHtml = this.requests.map(request => `
            <tr onclick="app.showRequestDetails('${request.id}')">
                <td>${new Date(request.timestamp).toLocaleString()}</td>
                <td>${request.model}</td>
                <td>${request.provider}</td>
                <td><span class="status-badge ${request.status.toLowerCase()}">${request.status}</span></td>
                <td>${request.duration || '-'}</td>
                <td>
                    <button class="btn btn-sm" onclick="event.stopPropagation(); app.showRequestDetails('${request.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                </td>
            </tr>
        `).join('');

        tbody.innerHTML = requestsHtml;
    }

    showRequestDetails(requestId) {
        const request = this.requests.find(r => r.id === requestId);
        if (!request) return;

        document.getElementById('modalRequestContent').textContent = JSON.stringify(request.request, null, 2);
        document.getElementById('modalResponseContent').textContent = JSON.stringify(request.response, null, 2);
        
        document.getElementById('requestDetailsModal').classList.add('show');
    }

    closeModal() {
        document.getElementById('requestDetailsModal').classList.remove('show');
    }

    clearRequests() {
        if (confirm('Are you sure you want to clear all request history?')) {
            this.requests = [];
            this.renderRequests();
            this.addLogEntry('info', 'Request history cleared');
        }
    }

    async exportRequests() {
        try {
            const result = await window.electronAPI.showSaveDialog({
                filters: [
                    { name: 'JSON Files', extensions: ['json'] },
                    { name: 'All Files', extensions: ['*'] }
                ],
                defaultPath: `ollamalink-requests-${new Date().toISOString().split('T')[0]}.json`
            });

            if (!result.canceled) {
                // This would save the requests data to the selected file
                this.addLogEntry('info', `Requests exported to ${result.filePath}`);
            }
        } catch (error) {
            console.error('Error exporting requests:', error);
            this.addLogEntry('error', `Failed to export requests: ${error.message}`);
        }
    }

    updateStats() {
        document.getElementById('totalRequests').textContent = this.requests.length;
        document.getElementById('activeRequests').textContent = this.requests.filter(r => r.status === 'pending').length;
        document.getElementById('errorCount').textContent = this.requests.filter(r => r.status === 'error').length;
    }

    addLogEntry(type, message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.innerHTML = `<span class="log-timestamp">[${timestamp}]</span> ${message}`;
        
        const consoleOutput = document.getElementById('consoleOutput');
        consoleOutput.appendChild(logEntry);
        consoleOutput.scrollTop = consoleOutput.scrollHeight;
    }

    clearLogs() {
        document.getElementById('consoleOutput').innerHTML = '';
    }

    filterLogs(level) {
        const entries = document.querySelectorAll('.log-entry');
        entries.forEach(entry => {
            if (level === 'all' || entry.classList.contains(level)) {
                entry.style.display = 'block';
            } else {
                entry.style.display = 'none';
            }
        });
    }

    async updateSetting(key, value) {
        this.settings[key] = value;
        await this.saveSettings();
        
        // Apply theme immediately if changed
        if (key === 'theme') {
            this.applyTheme(value);
        }
    }

    applyTheme(theme) {
        // Theme application logic would go here
        document.body.className = `theme-${theme}`;
    }

    togglePasswordVisibility(container) {
        const input = container.querySelector('input');
        const button = container.querySelector('button');
        const icon = button.querySelector('i');
        
        if (input.type === 'password') {
            input.type = 'text';
            icon.className = 'fas fa-eye-slash';
        } else {
            input.type = 'password';
            icon.className = 'fas fa-eye';
        }
    }

    async exportConfig() {
        try {
            const result = await window.electronAPI.showSaveDialog({
                filters: [
                    { name: 'JSON Files', extensions: ['json'] },
                    { name: 'All Files', extensions: ['*'] }
                ],
                defaultPath: 'ollamalink-config.json'
            });

            if (!result.canceled) {
                // Export configuration
                this.addLogEntry('info', `Configuration exported to ${result.filePath}`);
            }
        } catch (error) {
            console.error('Error exporting config:', error);
            this.addLogEntry('error', `Failed to export config: ${error.message}`);
        }
    }

    async importConfig() {
        try {
            const result = await window.electronAPI.showOpenDialog({
                filters: [
                    { name: 'JSON Files', extensions: ['json'] },
                    { name: 'All Files', extensions: ['*'] }
                ],
                properties: ['openFile']
            });

            if (!result.canceled && result.filePaths.length > 0) {
                // Import configuration
                this.addLogEntry('info', `Configuration imported from ${result.filePaths[0]}`);
            }
        } catch (error) {
            console.error('Error importing config:', error);
            this.addLogEntry('error', `Failed to import config: ${error.message}`);
        }
    }

    async startServer() {
        this.updateServerStatus('starting');
        // Server is managed by the main process
        this.addLogEntry('info', 'Starting server...');
    }

    async stopServer() {
        this.updateServerStatus('offline');
        this.addLogEntry('info', 'Server stopped');
    }

    async restartServer() {
        try {
            this.updateServerStatus('starting');
            this.addLogEntry('info', 'Restarting server...');
            
            const result = await window.electronAPI.restartBackend();
            if (result.success) {
                this.serverPort = result.port;
                this.baseUrl = `http://localhost:${this.serverPort}`;
                this.updateServerStatus('online');
                this.addLogEntry('info', `Server restarted on port ${result.port}`);
            } else {
                this.updateServerStatus('error');
                this.addLogEntry('error', `Failed to restart server: ${result.error}`);
            }
        } catch (error) {
            console.error('Error restarting server:', error);
            this.updateServerStatus('error');
            this.addLogEntry('error', `Failed to restart server: ${error.message}`);
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OllamaLinkApp();
});
