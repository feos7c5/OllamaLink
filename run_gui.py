import sys
import json
import logging
import time
import datetime
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QLineEdit,
    QComboBox, QCheckBox, QGroupBox, QSplitter, QGridLayout, QHeaderView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QFont, QTextCursor, QPixmap, QColor

# Import OllamaLink components
from core.router import OllamaRouter
from core.util import load_config, start_cloudflared_tunnel, is_cloudflared_installed, get_cloudflared_install_instructions
from core.request import OllamaRequestHandler
from core.response import OllamaResponseHandler
import core.api as api

# We need these FastAPI imports because the core API module is built on FastAPI
# Ideally, these would be encapsulated entirely within the core module
import uvicorn
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
import asyncio

# Set up logging
class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.text_edit.setReadOnly(True)
        # Use Arial font directly
        self.text_edit.setFont(QFont("Arial", 9))
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                          datefmt='%H:%M:%S')

    def emit(self, record):
        msg = self.formatter.format(record)
        # Apply color based on log level
        if record.levelname == 'ERROR':
            self.text_edit.append(f'<span style="color: #f44336;">{msg}</span>')
        elif record.levelname == 'WARNING':
            self.text_edit.append(f'<span style="color: #FF9800;">{msg}</span>')
        elif record.levelname == 'INFO':
            self.text_edit.append(f'<span style="color: #2196F3;">{msg}</span>')
        else:
            self.text_edit.append(msg)
        
        # Scroll to bottom
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        
        # Force processing events to ensure the UI updates immediately
        QApplication.processEvents()


class RequestLogEntry:
    """Represents a captured request and response"""
    def __init__(self, request_data, response_data=None, timestamp=None):
        self.request_data = request_data
        self.response_data = response_data
        self.timestamp = timestamp or int(time.time())
        self.model = request_data.get("model", "unknown") if request_data else "unknown"
        self.messages = request_data.get("messages", []) if request_data else []
        self.is_streaming = request_data.get("stream", False) if request_data else False
        self.completed_chunks = 0
        
        # Set initial status
        if response_data:
            if "error" in response_data:
                self.status = "Error"
            else:
                self.status = "Completed"
        else:
            self.status = "Pending"

class RequestLogger:
    """Handles logging of requests and responses for the GUI"""
    def __init__(self, update_signal=None):
        self.request_log = []
        self.update_signal = update_signal
        
    def log_request(self, request_data):
        """Log a new request"""
        entry = RequestLogEntry(request_data)
        self.request_log.append(entry)
        if self.update_signal:
            self.update_signal.emit(f"Request: {json.dumps(request_data, indent=2)}")
        return entry
        
    def log_response(self, request_data, response_data):
        """Log a response to a previous request"""
        # Find the matching request
        for entry in self.request_log:
            if entry.request_data == request_data:
                entry.response_data = response_data
                entry.status = "Completed" if "error" not in response_data else "Error"
                if self.update_signal:
                    self.update_signal.emit(f"Response: {json.dumps(response_data, indent=2)}")
                return entry
        return None
        
    def update_streaming_status(self, request_data, chunk_count, done=False):
        """Update status for a streaming request"""
        for entry in self.request_log:
            if entry.request_data == request_data:
                # Only log the update if it's a significant change or completion
                should_log = done or entry.completed_chunks == 0 or chunk_count % 10 == 0 or chunk_count - entry.completed_chunks >= 5
                
                # Update entry
                entry.completed_chunks = chunk_count
                if done:
                    entry.status = "Completed"
                    if self.update_signal:
                        self.update_signal.emit(f"Streaming request completed with {chunk_count} chunks")
                elif should_log and self.update_signal:
                    # Log progress updates at intervals
                    self.update_signal.emit(f"Streaming progress: {chunk_count} chunks received")
                
                return entry
        return None
        
    def log_error(self, request_data, error_message):
        """Log an error for a request"""
        for entry in self.request_log:
            if entry.request_data == request_data:
                entry.status = "Error"
                if self.update_signal:
                    self.update_signal.emit(f"Error: {error_message}")
                return entry
        return None
        
    def clear(self):
        """Clear all request logs"""
        self.request_log = []

class ServerThread(QThread):
    """Thread to run the uvicorn server"""
    update_signal = pyqtSignal(str)
    tunnel_url_signal = pyqtSignal(str)
    server_error_signal = pyqtSignal(str)
    
    def __init__(self, port, host, ollama_endpoint, use_tunnel, router=None, api_key=None):
        super().__init__()
        self.port = port
        self.host = host
        self.ollama_endpoint = ollama_endpoint
        self.use_tunnel = use_tunnel
        self.router = router
        self.api_key = api_key
        self.app = None
        self.server = None
        self.running = False
        
        # Initialize request logger for GUI integration
        self.request_logger = RequestLogger(self.update_signal)
        
        # Initialize core components
        self.response_handler = None
        self.request_handler = None
        
    def run(self):
        # Create custom update handler to show in GUI logs
        log_handler = logging.getLogger("ollamalink")
        def update_log(message):
            self.update_signal.emit(message)
            log_handler.info(message)
        
        update_log("Initializing OllamaLink server...")
        
        try:
            # Create the custom app with our core API module
            update_log("Creating API with integrated GUI request logger")
            self.app = api.create_api(
                ollama_endpoint=self.ollama_endpoint,
                api_key=self.api_key,
                request_logger=self.request_logger  # Pass the request logger to core API
            )
            
            # Get the core handlers from the app (they're created by the core API module)
            # We access them via hidden state since our core API already creates and uses them
            for route in self.app.routes:
                if hasattr(route, "endpoint") and route.path == "/v1/chat/completions":
                    if hasattr(route.endpoint, "__closure__"):
                        for cell in route.endpoint.__closure__:
                            if hasattr(cell, "cell_contents"):
                                if isinstance(cell.cell_contents, OllamaResponseHandler):
                                    self.response_handler = cell.cell_contents
                                elif isinstance(cell.cell_contents, OllamaRequestHandler):
                                    self.request_handler = cell.cell_contents
            
            # If we couldn't find the handlers in the existing app, create our own
            if not self.response_handler:
                self.response_handler = OllamaResponseHandler()
                update_log("Created new response handler")
            if not self.request_handler:
                self.request_handler = OllamaRequestHandler(
                    ollama_endpoint=self.ollama_endpoint, 
                    response_handler=self.response_handler
                )
                update_log("Created new request handler")
            
            # Configure and start the uvicorn server
            update_log(f"Starting server on {self.host}:{self.port}")
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning"
            )
            self.server = uvicorn.Server(config)
            self.running = True
            
            # Start tunnel if needed
            if self.use_tunnel:
                # Create a separate task to run the tunnel
                async def start_gui_cloudflared_tunnel():
                    try:
                        # Check if cloudflared is installed
                        if not is_cloudflared_installed():
                            self.update_signal.emit("cloudflared not found. Install instructions:")
                            self.update_signal.emit(get_cloudflared_install_instructions())
                            return

                        self.update_signal.emit("Starting cloudflared tunnel...")
                        
                        # Custom callback for GUI to update the UI
                        def tunnel_url_callback(url):
                            self.update_signal.emit(f"Tunnel started at: {url}")
                            self.tunnel_url_signal.emit(url)
                        
                        result = await start_cloudflared_tunnel(self.port, callback=tunnel_url_callback)
                        
                        if not result:
                            self.update_signal.emit("Could not get tunnel URL within timeout period.")
                            return
                        
                        tunnel_url, process = result
                        
                        # Keep the process running as long as the server is
                        watchdog_time = time.time()
                        while self.running:
                            # Check if process is still alive
                            if process.poll() is not None:
                                self.update_signal.emit("CloudFlare tunnel process terminated unexpectedly")
                                # Try to restart if we're still running
                                if self.running:
                                    self.update_signal.emit("Attempting to restart CloudFlare tunnel...")
                                    result = await start_cloudflared_tunnel(self.port, callback=tunnel_url_callback)
                                    if result:
                                        tunnel_url, process = result
                            
                            # Periodic keepalive log to show tunnel is still active
                            current_time = time.time()
                            if current_time - watchdog_time > 60:  # Log every minute
                                self.update_signal.emit("CloudFlare tunnel watchdog: still active")
                                watchdog_time = current_time
                                
                            await asyncio.sleep(5)
                        
                        # Cleanup process when done
                        if process and process.poll() is None:
                            self.update_signal.emit("Stopping CloudFlare tunnel...")
                            # Try graceful termination first
                            try:
                                process.terminate()
                                try:
                                    process.wait(timeout=5)
                                except subprocess.TimeoutExpired:
                                    self.update_signal.emit("Tunnel not responding to termination, forcing kill...")
                                    process.kill()
                            except Exception as e:
                                self.update_signal.emit(f"Error stopping tunnel: {str(e)}")
                        
                    except Exception as e:
                        self.update_signal.emit(f"Error with tunnel: {str(e)}")
                
                # Run both the server and tunnel
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.create_task(start_gui_cloudflared_tunnel())
                    loop.run_until_complete(self.server.serve())
                except Exception as e:
                    update_log(f"Error running server with tunnel: {str(e)}")
                    self.update_signal.emit(f"Error details: {type(e).__name__}: {str(e)}")
                    self.server_error_signal.emit("Server stopped due to an error")
            else:
                # Just run the server without tunnel
                try:
                    asyncio.run(self.server.serve())
                except Exception as e:
                    update_log(f"Error running server: {str(e)}")
                    self.update_signal.emit(f"Error details: {type(e).__name__}: {str(e)}")
                    self.server_error_signal.emit("Server stopped due to an error")
        
        except Exception as e:
            update_log(f"Error initializing server: {str(e)}")
            self.update_signal.emit(f"Error details: {type(e).__name__}: {str(e)}")
            self.server_error_signal.emit("Server failed to start")
            # Set running to False to indicate server is not actually running
            self.running = False
    
    def stop(self):
        """Safely stop the server thread"""
        self.running = False
        # The actual server will be stopped in the run method
        # This just signals that we should stop
    
    @property
    def request_log(self):
        """Access to the request logger's log entries"""
        return self.request_logger.request_log
    
    def clear_request_log(self):
        """Clear the request log"""
        self.request_logger.clear()

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self) 
        self.setWindowTitle("OllamaLink Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.server_thread = None
        self.router = None
        self.config = load_config()
        
        # Set application style
        self.set_app_style()
        
        # Set up UI
        self.setup_ui()
        
        # Initialize router
        self.init_router()
        
        # Update model display
        self.update_model_display()
        
        # Load model mappings in settings
        self.load_mappings()
    
    
    def set_app_style(self):
        """Set global application style"""
        # Modern style sheet for the entire application
        style = """
        QMainWindow, QWidget {
            background-color: #f9f9f9;
            color: #333333;
        }
        QTabWidget::pane {
            border: 1px solid #dddddd;
            border-radius: 4px;
            background-color: white;
        }
        QTabBar::tab {
            background-color: #f2f2f2;
            border: 1px solid #dddddd;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 8px 12px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            border-bottom: 1px solid white;
        }
        QGroupBox {
            border: 1px solid #dddddd;
            border-radius: 4px;
            margin-top: 10px;
            padding-top: 16px;
        }
        QLineEdit, QTextEdit, QComboBox {
            border: 1px solid #dddddd;
            border-radius: 4px;
            padding: 6px;
            background-color: white;
        }
        QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
            border: 1px solid #2196F3;
        }
        QPushButton {
            border-radius: 4px;
            padding: 6px 12px;
        }
        QTableWidget {
            border: 1px solid #dddddd;
            border-radius: 4px;
            gridline-color: #f2f2f2;
        }
        QHeaderView::section {
            background-color: #f2f2f2;
            border: 1px solid #dddddd;
            padding: 6px;
        }
        """
        self.setStyleSheet(style)
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Create tabs
        dashboard_tab = QWidget()
        console_tab = QWidget()
        requests_tab = QWidget()
        settings_tab = QWidget()
        
        # Add tabs
        tabs.addTab(dashboard_tab, "Dashboard")
        tabs.addTab(console_tab, "Console")
        tabs.addTab(requests_tab, "Requests/Responses")
        tabs.addTab(settings_tab, "Settings")
        
        # Connect tab change event
        tabs.currentChanged.connect(self.on_tab_changed)
        
        # Set up individual tabs
        self.setup_dashboard_tab(dashboard_tab)
        self.setup_console_tab(console_tab)
        self.setup_requests_tab(requests_tab)
        self.setup_settings_tab(settings_tab)
        
        # Status bar
        self.status_label = QLabel("Not Running")
        self.statusBar().addWidget(self.status_label)
    
    def on_tab_changed(self, index):
        """Handle tab change events"""
        # If we're switching to the Settings tab, reload the mappings
        if index == 3:  # Settings tab
            self.load_mappings()
            
        # If we're switching to the Dashboard tab, update the model display
        if index == 0:  # Dashboard tab
            self.update_model_display()
    
    def setup_dashboard_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Create top section with server controls and status
        top_section = QHBoxLayout()
        
        # Server control section
        server_group = QGroupBox("Server Control")
        server_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; padding-top: 16px; }")
        server_layout = QVBoxLayout()
        server_group.setLayout(server_layout)
        
        buttons_layout = QHBoxLayout()
        
        # Start/Stop buttons with styling
        self.start_button = QPushButton("Start Server")
        self.start_button.clicked.connect(self.start_server)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-weight: bold; border-radius: 4px;")
        self.start_button.setMinimumWidth(120)
        
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.clicked.connect(self.stop_server)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; font-weight: bold; border-radius: 4px;")
        self.stop_button.setMinimumWidth(120)
        self.stop_button.setEnabled(False)
        
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        
        server_layout.addLayout(buttons_layout)
        
        # Server status
        status_group = QGroupBox("Server Status")
        status_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; padding-top: 16px; }")
        status_layout = QGridLayout()  # Use grid layout for better alignment
        status_group.setLayout(status_layout)
        
        # Create labels with fixed width for alignment
        status_label = QLabel("<b>Status:</b>")
        port_label = QLabel("<b>Port:</b>")
        local_url_label = QLabel("<b>Local URL:</b>")
        cf_url_label = QLabel("<b>CloudFlare URL:</b>")
        
        # Set fixed width for labels to ensure alignment
        label_width = 120
        status_label.setFixedWidth(label_width)
        port_label.setFixedWidth(label_width)
        local_url_label.setFixedWidth(label_width)
        cf_url_label.setFixedWidth(label_width)
        
        # Create value widgets
        self.status_field = QLabel("Not Running")
        self.status_field.setStyleSheet("font-weight: bold; color: #f44336;")
        self.port_field = QLabel(str(self.config["server"]["port"]))
        
        # Local URL with layout
        local_url_layout = QHBoxLayout()
        local_url_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.url_field = QLabel("-")
        self.url_field.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        local_url_layout.addWidget(self.url_field, 1)  # Add stretch factor
        local_url_layout.addStretch(0)  # Remove extra stretch
        
        # CloudFlare URL with copy button
        cf_url_layout = QHBoxLayout()
        cf_url_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.tunnel_url_field = QLabel("-")
        self.tunnel_url_field.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tunnel_url_field.setStyleSheet("font-weight: bold;")
        
        # Copy button for CloudFlare URL
        self.copy_cf_button = QPushButton("Copy")
        self.copy_cf_button.setStyleSheet("background-color: #2196F3; color: white; padding: 4px; border-radius: 4px;")
        self.copy_cf_button.setFixedWidth(50)
        self.copy_cf_button.clicked.connect(self.copy_cloudflare_url)
        self.copy_cf_button.setEnabled(False)
        
        cf_url_layout.addWidget(self.tunnel_url_field, 1)  # Add stretch factor
        cf_url_layout.addWidget(self.copy_cf_button, 0)  # No stretch
        
        # Add fields to grid layout - use 4 rows x 2 columns
        status_layout.addWidget(status_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.status_field, 0, 1, Qt.AlignmentFlag.AlignLeft)
        
        status_layout.addWidget(port_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.port_field, 1, 1, Qt.AlignmentFlag.AlignLeft)
        
        status_layout.addWidget(local_url_label, 2, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addLayout(local_url_layout, 2, 1)
        
        status_layout.addWidget(cf_url_label, 3, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addLayout(cf_url_layout, 3, 1)
        
        # Stretch the second column
        status_layout.setColumnStretch(1, 1)
        
        # Add to top section
        top_section.addWidget(server_group)
        top_section.addWidget(status_group)
        
        # Models section with better styling
        models_section = QHBoxLayout()
        
        # Model mapping table
        models_group = QGroupBox("Model Mapping")
        models_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; padding-top: 16px; }")
        models_layout = QVBoxLayout()
        models_group.setLayout(models_layout)
        
        # Header label
        header_label = QLabel("API Model → Ollama Model")
        header_label.setStyleSheet("font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        models_layout.addWidget(header_label)
        
        # Model mapping table
        self.model_table = QTableWidget(0, 3)
        self.model_table.setHorizontalHeaderLabels(["API Model", "Ollama Model", "Resolved Model"])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setStyleSheet("alternate-background-color: #f2f2f2; border: 1px solid #ddd; border-radius: 4px;")
        self.model_table.horizontalHeader().setStyleSheet("font-weight: bold; background-color: #e0e0e0;")
        
        # Set column widths
        header = self.model_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
        models_layout.addWidget(self.model_table)
        
        # Available models section
        available_group = QGroupBox("Available Models")
        available_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; padding-top: 16px; }")
        available_layout = QVBoxLayout()
        available_group.setLayout(available_layout)
        
        # Header label
        available_header = QLabel("Ollama Models")
        available_header.setStyleSheet("font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        available_layout.addWidget(available_header)
        
        self.available_models_text = QTextEdit()
        self.available_models_text.setReadOnly(True)
        # Use Arial font directly
        self.available_models_text.setFont(QFont("Arial", 10))
        self.available_models_text.setStyleSheet("border: 1px solid #ddd; border-radius: 4px;")
        
        available_layout.addWidget(self.available_models_text)
        
        # Add to models section
        models_section.addWidget(models_group, 3)  # 60% width
        models_section.addWidget(available_group, 2)  # 40% width
        
        # Add sections to main layout
        layout.addLayout(top_section)
        layout.addLayout(models_section, 1)  # Give this section more vertical space
    
    def setup_console_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Header
        header = QLabel("Server Console Output")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Console container group
        console_group = QGroupBox()
        console_group.setStyleSheet("QGroupBox { border: 1px solid #ddd; border-radius: 4px; margin-top: 5px; }")
        console_layout = QVBoxLayout()
        console_group.setLayout(console_layout)
        
        # Log viewer with better styling
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Use Arial font directly
        self.log_text.setFont(QFont("Arial", 9))
        self.log_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 8px;")
        self.log_text.setMinimumHeight(400)
        self.log_text.setAcceptRichText(True)  # Enable rich text for color formatting
        console_layout.addWidget(self.log_text)
        
        # Add the console group to main layout
        layout.addWidget(console_group)
        
        # Controls bar
        controls_layout = QHBoxLayout()
        
        # Level filter combobox
        level_label = QLabel("Log Level:")
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["ALL", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentIndex(0)
        self.log_level_combo.currentIndexChanged.connect(self.set_log_level)
        self.log_level_combo.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 4px;")
        
        # Add a spacer
        controls_layout.addWidget(level_label)
        controls_layout.addWidget(self.log_level_combo)
        controls_layout.addStretch(1)
        
        # Clear button with styling
        clear_button = QPushButton("Clear Log")
        clear_button.clicked.connect(self.log_text.clear)
        clear_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        clear_button.setMinimumWidth(120)
        controls_layout.addWidget(clear_button)
        
        # Add controls to layout with padding
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)
        controls_widget.setStyleSheet("margin-top: 10px;")
        layout.addWidget(controls_widget)
        
        # Set up logging to the text edit
        log_handler = QTextEditLogger(self.log_text)
        log_handler.setLevel(logging.INFO)
        
        # Remove any existing handlers of the same type to avoid duplicates
        for handler in logging.getLogger().handlers:
            if isinstance(handler, QTextEditLogger):
                logging.getLogger().removeHandler(handler)
                
        # Add our handler
        logging.getLogger().addHandler(log_handler)
        
        # Set a shorter formatter to avoid long timestamps in GUI
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                      datefmt='%H:%M:%S')
        log_handler.setFormatter(formatter)
        
        # Set up a timer to periodically refresh the logs
        self.log_refresh_timer = QTimer()
        self.log_refresh_timer.timeout.connect(lambda: QApplication.processEvents())
        self.log_refresh_timer.start(500)  # Refresh every 500ms
    
    def set_log_level(self, index):
        """Set the log level filter"""
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]
        if 0 <= index < len(levels):
            for handler in logging.getLogger().handlers:
                if isinstance(handler, QTextEditLogger):
                    handler.setLevel(levels[index])
                    logging.info(f"Log level set to {self.log_level_combo.currentText()}")
    
    def setup_requests_tab(self, tab):
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Header
        header = QLabel("API Requests & Responses")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Request/Response view
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Requests list section
        requests_widget = QWidget()
        requests_layout = QVBoxLayout()
        requests_widget.setLayout(requests_layout)
        
        # Header with auto-refresh toggle
        header_layout = QHBoxLayout()
        
        requests_header = QLabel("Request History")
        requests_header.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(requests_header)
        
        header_layout.addStretch(1)
        
        # Auto-refresh checkbox
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
        self.auto_refresh_checkbox.setChecked(True)
        header_layout.addWidget(self.auto_refresh_checkbox)
        
        requests_layout.addLayout(header_layout)
        
        # Requests list with better styling
        self.request_table = QTableWidget(0, 4)
        self.request_table.setHorizontalHeaderLabels(["Timestamp", "Model", "Messages", "Status"])
        self.request_table.horizontalHeader().setStretchLastSection(True)
        self.request_table.setAlternatingRowColors(True)
        self.request_table.setStyleSheet("alternate-background-color: #f2f2f2;")
        self.request_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.request_table.itemClicked.connect(self.on_request_selected)
        self.request_table.setMinimumHeight(200)  # Set minimum height for request table
        
        requests_layout.addWidget(self.request_table)
        
        # Request/Response detail section
        detail_widget = QWidget()
        detail_layout = QVBoxLayout()
        detail_widget.setLayout(detail_layout)
        
        # Request section
        request_group = QGroupBox("Request Details")
        request_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        request_group_layout = QVBoxLayout()
        request_group.setLayout(request_group_layout)
        
        self.request_text = QTextEdit()
        self.request_text.setReadOnly(True)
        # Use Arial font directly
        self.request_text.setFont(QFont("Arial", 9))
        self.request_text.setStyleSheet("background-color: #f8f8f8;")
        
        request_group_layout.addWidget(self.request_text)
        
        # Response section
        response_group = QGroupBox("Response Details")
        response_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        response_group_layout = QVBoxLayout()
        response_group.setLayout(response_group_layout)
        
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        # Use Arial font directly
        self.response_text.setFont(QFont("Arial", 9))
        self.response_text.setStyleSheet("background-color: #f8f8f8;")
        
        response_group_layout.addWidget(self.response_text)
        
        # Add to detail layout
        detail_layout.addWidget(request_group)
        detail_layout.addWidget(response_group)
        
        # Add to splitter
        splitter.addWidget(requests_widget)
        splitter.addWidget(detail_widget)
        
        # Initialize splitter sizes - give the request history more space
        splitter.setSizes([300, 500])  # Initial split: 300px for requests, 500px for details
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Controls bar
        buttons_layout = QHBoxLayout()
        
        # Refresh button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_request_history)
        refresh_button.setStyleSheet("background-color: #2196F3; color: white; padding: 8px;")
        refresh_button.setMinimumWidth(120)
        
        # Clear button
        clear_button = QPushButton("Clear History")
        clear_button.clicked.connect(self.clear_request_history)
        clear_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        clear_button.setMinimumWidth(120)
        
        buttons_layout.addWidget(refresh_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(clear_button)
        
        layout.addLayout(buttons_layout)
        
        # Auto-refresh timer
        self.request_refresh_timer = QTimer()
        self.request_refresh_timer.timeout.connect(self.refresh_request_history)
        self.request_refresh_timer.start(1000)  # Refresh every second
    
    def refresh_request_history(self):
        """Refresh the request history table"""
        if not self.auto_refresh_checkbox.isChecked():
            return
            
        # Store currently selected item
        selected_row = -1
        selected_items = self.request_table.selectedItems()
        if selected_items:
            selected_row = selected_items[0].row()
        
        # Update the table
        self.update_request_table()
        
        # Restore selection if possible
        if selected_row >= 0 and selected_row < self.request_table.rowCount():
            self.request_table.selectRow(selected_row)
            
    def update_request_table(self):
        """Update only the request table without logging"""
        # Only update if we have a server thread
        if not self.server_thread or not hasattr(self.server_thread, "request_log"):
            return
            
        # Clear the table
        self.request_table.setRowCount(0)
        
        # Add entries
        for i, entry in enumerate(self.server_thread.request_log):
            self.request_table.insertRow(i)
            
            # Convert timestamp
            dt = datetime.datetime.fromtimestamp(entry.timestamp)
            time_str = dt.strftime("%H:%M:%S")
            
            # Add data
            self.request_table.setItem(i, 0, QTableWidgetItem(time_str))
            self.request_table.setItem(i, 1, QTableWidgetItem(entry.model))
            
            # Count messages
            msg_count = len(entry.messages)
            self.request_table.setItem(i, 2, QTableWidgetItem(f"{msg_count} messages"))
            
            # Status with color and streaming indication
            status_text = entry.status
            if entry.is_streaming and entry.status == "Pending" and hasattr(entry, 'completed_chunks') and entry.completed_chunks > 0:
                status_text = f"Streaming ({entry.completed_chunks})"
            
            status_item = QTableWidgetItem(status_text)
            
            # Set color based on status
            if entry.status == "Completed":
                status_item.setForeground(QColor("#4CAF50"))  # Green
            elif entry.status == "Timeout":
                status_item.setForeground(QColor("#FF9800"))  # Orange
            elif entry.status == "Error":
                status_item.setForeground(QColor("#F44336"))  # Red
            elif status_text.startswith("Streaming"):
                status_item.setForeground(QColor("#2196F3"))  # Blue for active streaming
            
            self.request_table.setItem(i, 3, status_item)
        
        # Resize columns
        self.request_table.resizeColumnsToContents()

    def update_request_log(self, message):
        """Update the request log with a new message"""
        logging.info(message)
        
        # Update the request table
        self.update_request_table()
        
        # Force update of the UI
        QApplication.processEvents()
    
    def setup_settings_tab(self, tab):
        layout = QHBoxLayout()  # Change to horizontal layout
        tab.setLayout(layout)
        
        # Left side - Settings
        settings_group = QGroupBox("Server Settings")
        settings_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; padding-top: 16px; }")
        settings_layout = QGridLayout()  # Use grid layout for better alignment
        settings_group.setLayout(settings_layout)
        
        # Create labels with fixed width
        host_label = QLabel("Host:")
        port_label = QLabel("Port:")
        tunnel_label = QLabel("Use Tunnel:")
        ollama_label = QLabel("Ollama Endpoint:")
        openai_key_label = QLabel("OpenAI API Key:")
        
        # Set fixed width for labels
        label_width = 120
        host_label.setFixedWidth(label_width)
        port_label.setFixedWidth(label_width)
        tunnel_label.setFixedWidth(label_width)
        ollama_label.setFixedWidth(label_width)
        openai_key_label.setFixedWidth(label_width)
        
        # Server settings
        self.host_input = QLineEdit(self.config["server"]["hostname"])
        self.host_input.setStyleSheet("padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        
        self.port_input = QLineEdit(str(self.config["server"]["port"]))
        self.port_input.setStyleSheet("padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        
        self.tunnel_checkbox = QCheckBox("Use Cloudflared Tunnel")
        self.tunnel_checkbox.setChecked(self.config["cloudflared"]["use_tunnel"])
        
        # Ollama settings
        self.ollama_endpoint_input = QLineEdit(self.config["ollama"]["endpoint"])
        self.ollama_endpoint_input.setStyleSheet("padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        
        # OpenAI API Key settings
        self.openai_api_key_input = QLineEdit()
        self.openai_api_key_input.setStyleSheet("padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        self.openai_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)  # Mask the API key for security
        # Set the API key if it exists in the config
        if "openai" in self.config and "api_key" in self.config["openai"]:
            self.openai_api_key_input.setText(self.config["openai"]["api_key"])
        
        # Create a layout for the API key input with toggle button
        api_key_layout = QHBoxLayout()
        api_key_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        api_key_layout.addWidget(self.openai_api_key_input)
        
        # Add toggle button for API key visibility
        self.api_key_toggle_button = QPushButton("🔒")  # Initial state showing the "show" icon
        self.api_key_toggle_button.setStyleSheet("padding: 4px 8px; border: 1px solid #ddd; border-radius: 4px;")
        self.api_key_toggle_button.setFixedWidth(40)
        self.api_key_toggle_button.setToolTip("Show API Key")
        self.api_key_toggle_button.clicked.connect(self.toggle_api_key_visibility)
        api_key_layout.addWidget(self.api_key_toggle_button)
        
        # Add to layout
        settings_layout.addWidget(host_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.host_input, 0, 1)
        
        settings_layout.addWidget(port_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.port_input, 1, 1)
        
        settings_layout.addWidget(tunnel_label, 2, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.tunnel_checkbox, 2, 1)
        
        settings_layout.addWidget(ollama_label, 3, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.ollama_endpoint_input, 3, 1)
        
        settings_layout.addWidget(openai_key_label, 4, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addLayout(api_key_layout, 4, 1)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        save_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        save_button.setMinimumWidth(120)
        
        # Add spacer and button at bottom
        settings_layout.addWidget(QWidget(), 5, 0, 1, 2)  # Empty spacer
        settings_layout.addWidget(save_button, 6, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        settings_layout.setRowStretch(5, 1)  # Make the spacer expand
        
        # Right side - Model Mappings
        mappings_group = QGroupBox("Model Mappings")
        mappings_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; padding-top: 16px; }")
        mappings_layout = QVBoxLayout()
        mappings_group.setLayout(mappings_layout)
        
        # Existing mappings table with header
        mapping_header = QLabel("Current Model Mappings:")
        mapping_header.setStyleSheet("font-weight: bold; color: #2196F3; margin-bottom: 5px;")
        mappings_layout.addWidget(mapping_header)
        
        # Mappings table
        self.mappings_table = QTableWidget(0, 2)
        self.mappings_table.setHorizontalHeaderLabels(["API Model", "Ollama Model"])
        self.mappings_table.horizontalHeader().setStretchLastSection(True)
        self.mappings_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.mappings_table.setAlternatingRowColors(True)
        self.mappings_table.setStyleSheet("alternate-background-color: #f2f2f2; border: 1px solid #ddd; border-radius: 4px;")
        self.mappings_table.horizontalHeader().setStyleSheet("font-weight: bold; background-color: #e0e0e0;")
        
        # Set column widths
        header = self.mappings_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        # Add new mapping section
        mapping_form = QGroupBox("Add New Mapping")
        mapping_form.setStyleSheet("QGroupBox { border: 1px solid #ddd; border-radius: 4px; margin-top: 5px; padding-top: 16px; }")
        mapping_form_layout = QGridLayout()
        mapping_form.setLayout(mapping_form_layout)
        
        # Create labels
        api_label = QLabel("API Model Name:")
        ollama_label = QLabel("Ollama Model:")
        
        self.new_api_model = QLineEdit()
        self.new_api_model.setPlaceholderText("e.g., gpt-4o, claude-3-opus")
        self.new_api_model.setStyleSheet("padding: 6px; border: 1px solid #ddd; border-radius: 4px;")
        
        self.new_ollama_model = QComboBox()
        self.new_ollama_model.setEditable(True)
        self.new_ollama_model.setPlaceholderText("Select or type Ollama model")
        self.new_ollama_model.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 4px;")
        
        # Add items to form
        mapping_form_layout.addWidget(api_label, 0, 0)
        mapping_form_layout.addWidget(self.new_api_model, 0, 1)
        mapping_form_layout.addWidget(ollama_label, 1, 0)
        mapping_form_layout.addWidget(self.new_ollama_model, 1, 1)
        
        # Default model section
        default_group = QGroupBox("Default Model")
        default_group.setStyleSheet("QGroupBox { border: 1px solid #ddd; border-radius: 4px; margin-top: 5px; padding-top: 16px; }")
        default_layout = QHBoxLayout()
        default_group.setLayout(default_layout)
        
        default_label = QLabel("Default Model:")
        self.default_model_combo = QComboBox()
        self.default_model_combo.setStyleSheet("padding: 5px; border: 1px solid #ddd; border-radius: 4px;")
        default_layout.addWidget(default_label)
        default_layout.addWidget(self.default_model_combo, 1)
        
        # Add/Remove buttons
        buttons_layout = QHBoxLayout()
        
        add_button = QPushButton("Add Mapping")
        add_button.clicked.connect(self.add_mapping)
        add_button.setStyleSheet("background-color: #2196F3; color: white; padding: 6px; font-weight: bold; border-radius: 4px;")
        add_button.setMinimumWidth(120)
        
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_mapping)
        remove_button.setStyleSheet("background-color: #f44336; color: white; padding: 6px; font-weight: bold; border-radius: 4px;")
        remove_button.setMinimumWidth(120)
        
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(add_button)
        buttons_layout.addWidget(remove_button)
        buttons_layout.addStretch(1)
        
        # Add to mappings layout
        mappings_layout.addWidget(self.mappings_table)
        mappings_layout.addWidget(mapping_form)
        mappings_layout.addWidget(default_group)
        mappings_layout.addLayout(buttons_layout)
        mappings_layout.addStretch(1)
        
        # Add both groups to main layout
        layout.addWidget(settings_group, 1)
        layout.addWidget(mappings_group, 2)  # Give mappings more space
        
        # Load mappings
        self.load_mappings()
    
    def load_mappings(self):
        """Load model mappings from config to the UI"""
        # Clear tables
        self.mappings_table.setRowCount(0)
        self.new_ollama_model.clear()
        self.default_model_combo.clear()
        
        # Add available models to the dropdown
        if self.router and self.router.available_models:
            for model in self.router.available_models:
                self.new_ollama_model.addItem(model)
                self.default_model_combo.addItem(model)
        
        # Load existing mappings
        if "ollama" in self.config and "model_mappings" in self.config["ollama"]:
            mappings = self.config["ollama"]["model_mappings"]
            
            # Set default model if available
            if "default" in mappings:
                default_model = mappings["default"]
                index = self.default_model_combo.findText(default_model)
                if index >= 0:
                    self.default_model_combo.setCurrentIndex(index)
            
            # Add model mappings to table
            row = 0
            for api_model, ollama_model in mappings.items():
                if api_model == "default":
                    continue
                
                self.mappings_table.insertRow(row)
                self.mappings_table.setItem(row, 0, QTableWidgetItem(api_model))
                self.mappings_table.setItem(row, 1, QTableWidgetItem(ollama_model))
                row += 1
    
    def add_mapping(self):
        """Add a new model mapping"""
        api_model = self.new_api_model.text().strip()
        ollama_model = self.new_ollama_model.currentText().strip()
        
        if not api_model or not ollama_model:
            logging.warning("Both API model and Ollama model must be specified")
            return
        
        # Add to table
        row = self.mappings_table.rowCount()
        self.mappings_table.insertRow(row)
        self.mappings_table.setItem(row, 0, QTableWidgetItem(api_model))
        self.mappings_table.setItem(row, 1, QTableWidgetItem(ollama_model))
        
        # Clear inputs
        self.new_api_model.clear()
        self.new_ollama_model.setCurrentIndex(-1)
        
        # Save to config
        self.save_mappings()
        
        # Also update the dashboard view immediately
        self.update_model_display()
    
    def remove_mapping(self):
        """Remove selected mapping"""
        selected_rows = set()
        for item in self.mappings_table.selectedItems():
            selected_rows.add(item.row())
        
        # Remove in reverse order to avoid index issues
        for row in sorted(selected_rows, reverse=True):
            self.mappings_table.removeRow(row)
        
        # Save to config
        self.save_mappings()
        
        # Also update the dashboard view immediately
        self.update_model_display()
    
    def save_mappings(self):
        """Save model mappings from UI to config"""
        mappings = {}
        
        # Add default model
        default_model = self.default_model_combo.currentText()
        if default_model:
            mappings["default"] = default_model
        
        # Add other mappings
        for row in range(self.mappings_table.rowCount()):
            api_model = self.mappings_table.item(row, 0).text()
            ollama_model = self.mappings_table.item(row, 1).text()
            
            if api_model and ollama_model:
                mappings[api_model] = ollama_model
        
        # Update config
        if "ollama" not in self.config:
            self.config["ollama"] = {}
        
        self.config["ollama"]["model_mappings"] = mappings
        
        # Save to file
        try:
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Model mappings saved to config.json: {json.dumps(mappings)}")
            
            # Reinitialize router with new settings
            self.init_router()
            self.update_model_display()
        except Exception as e:
            logging.error(f"Error saving model mappings: {str(e)}")

    def save_settings(self):
        """Save settings to config.json"""
        self.config["server"]["hostname"] = self.host_input.text()
        self.config["server"]["port"] = int(self.port_input.text())
        self.config["cloudflared"]["use_tunnel"] = self.tunnel_checkbox.isChecked()
        self.config["ollama"]["endpoint"] = self.ollama_endpoint_input.text()
        
        # Save OpenAI API key
        api_key = self.openai_api_key_input.text().strip()
        if api_key:
            if "openai" not in self.config:
                self.config["openai"] = {}
            self.config["openai"]["api_key"] = api_key
        
        # Save model mappings too
        self.save_mappings()
        
        # Save to file
        try:
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info("Settings saved to config.json")
            
            # Reinitialize router with new settings
            self.init_router()
            self.update_model_display()
        except Exception as e:
            logging.error(f"Error saving settings: {str(e)}")
    
    def init_router(self):
        """Initialize the OllamaRouter with current settings"""
        try:
            ollama_endpoint = self.config["ollama"]["endpoint"]
            self.router = OllamaRouter(ollama_endpoint=ollama_endpoint)
            logging.info(f"Router initialized with endpoint: {ollama_endpoint}")
        except Exception as e:
            logging.error(f"Error initializing router: {str(e)}")
    
    def update_model_display(self):
        """Update the model tables with current router information"""
        if not self.router:
            return
        
        # Update available models
        self.available_models_text.clear()
        if hasattr(self.router, 'connection_error') and self.router.connection_error:
            self.available_models_text.append(f"Error connecting to Ollama: {self.router.connection_error}")
        else:
            self.available_models_text.append(f"<b>Default model:</b> {self.router.default_model}\n")
            self.available_models_text.append("<b>Available models:</b>")
            for model in self.router.available_models:
                self.available_models_text.append(f"• {model}")
        
        # Update model mapping table
        self.model_table.setRowCount(0)
        
        # Get mappings from router or config
        mappings = {}
        if hasattr(self.router, 'model_mappings') and self.router.model_mappings:
            mappings = self.router.model_mappings
        elif "ollama" in self.config and "model_mappings" in self.config["ollama"]:
            mappings = self.config["ollama"]["model_mappings"]
            
        # Add model mappings to table
        if mappings:
            row = 0
            for api_model, ollama_model in mappings.items():
                if api_model == "default":
                    continue
                
                self.model_table.insertRow(row)
                
                # Create styled items
                api_item = QTableWidgetItem(api_model)
                api_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                
                ollama_item = QTableWidgetItem(ollama_model)
                ollama_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                
                # Get the actual resolved model (if router is available)
                resolved_model = ""
                if hasattr(self.router, 'get_model_name'):
                    resolved_model = self.router.get_model_name(api_model)
                else:
                    resolved_model = ollama_model
                
                resolved_item = QTableWidgetItem(resolved_model)
                resolved_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                
                # Add items to table
                self.model_table.setItem(row, 0, api_item)
                self.model_table.setItem(row, 1, ollama_item)
                self.model_table.setItem(row, 2, resolved_item)
                
                row += 1
            
            # Resize rows to contents
            self.model_table.resizeRowsToContents()
    
    def copy_cloudflare_url(self):
        """Copy CloudFlare URL to clipboard"""
        url = self.tunnel_url_field.text()
        if url and url != "-" and url != "Disabled" and url != "Starting...":
            clipboard = QApplication.clipboard()
            clipboard.setText(url)
            logging.info(f"Copied to clipboard: {url}")
            
            # Show visual feedback
            original_text = self.copy_cf_button.text()
            original_style = self.copy_cf_button.styleSheet()
            
            self.copy_cf_button.setText("✓")
            self.copy_cf_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 4px;")
            
            # Reset after a short delay
            def reset_button():
                self.copy_cf_button.setText(original_text)
                self.copy_cf_button.setStyleSheet(original_style)
            
            QTimer.singleShot(1500, reset_button)

    def update_tunnel_url(self, url):
        """Update the CloudFlare tunnel URL"""
        if url:
            api_url = f"{url}/v1"
            self.tunnel_url_field.setText(api_url)
            self.copy_cf_button.setEnabled(True)
            logging.info(f"CloudFlare tunnel URL: {api_url}")
            
            # Update the status bar too
            local_url = self.url_field.text()
            self.status_label.setText(f"Running locally on {local_url} and via tunnel at {api_url}")
    
    def start_server(self):
        """Start the FastAPI server"""
        if self.server_thread and self.server_thread.running:
            return
        
        # Disable start button immediately to prevent double clicks
        self.start_button.setEnabled(False)
        
        host = self.host_input.text()
        port = int(self.port_input.text())
        ollama_endpoint = self.ollama_endpoint_input.text()
        use_tunnel = self.tunnel_checkbox.isChecked()
        
        # Get API key if available
        api_key = None
        if "openai" in self.config and "api_key" in self.config["openai"]:
            api_key = self.config["openai"]["api_key"]
        
        self.server_thread = ServerThread(
            port=port,
            host=host,
            ollama_endpoint=ollama_endpoint,
            use_tunnel=use_tunnel,
            router=self.router,
            api_key=api_key
        )
        
        # Connect signals
        self.server_thread.update_signal.connect(self.update_request_log)
        self.server_thread.tunnel_url_signal.connect(self.update_tunnel_url)
        self.server_thread.server_error_signal.connect(self.handle_server_error)
        
        # Start the server
        self.server_thread.start()
        
        # Update UI
        self.stop_button.setEnabled(True)
        self.status_field.setText("Running")
        self.status_field.setStyleSheet("font-weight: bold; color: #4CAF50;")
        self.port_field.setText(str(port))
        base_url = f"http://{host}:{port}/v1"
        self.url_field.setText(base_url)
        self.status_label.setText(f"Running on {base_url}")
        
        # Reset CloudFlare URL if not using tunnel
        if not use_tunnel:
            self.tunnel_url_field.setText("Disabled")
            self.copy_cf_button.setEnabled(False)
        else:
            self.tunnel_url_field.setText("Starting...")
            self.copy_cf_button.setEnabled(False)
        
        # Update the model mapping display to show current mappings
        self.update_model_display()
        
        logging.info(f"Server started on {base_url}")
    
    def stop_server(self):
        """Stop the server"""
        if self.server_thread and self.server_thread.running:
            logging.info("Stopping server...")
            self.status_field.setText("Stopping...")
            self.status_field.setStyleSheet("font-weight: bold; color: #FFA000;")
            
            # Disable both buttons while stopping
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            
            # Signal the thread to stop
            self.server_thread.running = False
            
            # Force terminate the thread if it doesn't stop within a timeout
            max_wait_time = 5000  # 5 seconds
            check_interval = 100  # 100ms
            checks_performed = 0
            max_checks = max_wait_time // check_interval
            
            def check_thread_stopped():
                nonlocal checks_performed
                checks_performed += 1
                
                if not self.server_thread.isRunning():
                    # Thread stopped normally, update UI
                    self.server_thread = None
                    self.update_ui_after_stop()
                    return
                
                if checks_performed >= max_checks:
                    # Thread didn't stop within timeout, force termination
                    logging.warning("Server thread didn't stop within timeout, forcing termination")
                    self.server_thread.terminate()
                    self.server_thread = None
                    self.update_ui_after_stop()
                    return
                
                # Check again after interval
                QTimer.singleShot(check_interval, check_thread_stopped)
            
            # Start checking if the thread has stopped
            QTimer.singleShot(check_interval, check_thread_stopped)
            
        else:
            # No server is running, just update UI
            self.update_ui_after_stop()

    def update_ui_after_stop(self):
        """Update UI elements after server has stopped"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_field.setText("Not Running")
        self.status_field.setStyleSheet("font-weight: bold; color: #f44336;")
        self.status_label.setText("Not Running")
        self.tunnel_url_field.setText("-")
        self.copy_cf_button.setEnabled(False)
        logging.info("Server stopped")
    
    def on_request_selected(self, item):
        """Handle request selection from the request table"""
        row = item.row()
        
        if self.server_thread and hasattr(self.server_thread, "request_log"):
            if row < len(self.server_thread.request_log):
                entry = self.server_thread.request_log[row]
                
                # Show request with formatting
                formatted_request = json.dumps(entry.request_data, indent=2)
                self.request_text.setText(formatted_request)
                
                # Show response if available
                if entry.response_data:
                    formatted_response = json.dumps(entry.response_data, indent=2)
                    
                    # Add color for error responses
                    if entry.status == "Error" and "error" in entry.response_data:
                        error_msg = entry.response_data["error"].get("message", "Unknown error")
                        self.response_text.setHtml(f'<span style="color: #F44336; font-weight: bold;">ERROR: {error_msg}</span><br><br>' + 
                                                 f'<pre>{formatted_response}</pre>')
                    else:
                        self.response_text.setText(formatted_response)
                else:
                    if entry.status == "Timeout":
                        self.response_text.setHtml('<span style="color: #FF9800; font-weight: bold;">Request timed out</span>')
                    elif entry.status == "Error":
                        self.response_text.setHtml('<span style="color: #F44336; font-weight: bold;">Error occurred</span>')
                    elif entry.is_streaming:
                        if entry.status == "Completed":
                            self.response_text.setHtml('<span style="color: #4CAF50; font-weight: bold;">Streaming request completed</span><br>' +
                                                   f'<span>Received {entry.completed_chunks} chunks</span>')
                        elif entry.completed_chunks > 0:
                            self.response_text.setHtml('<span style="color: #2196F3; font-weight: bold;">Streaming in progress</span><br>' +
                                                   f'<span>Received {entry.completed_chunks} chunks so far</span>')
                        else:
                            self.response_text.setText("Streaming request - waiting for data")
                    else:
                        self.response_text.setText("No response data available")

    def clear_request_history(self):
        """Clear the request history"""
        if self.server_thread:
            self.server_thread.clear_request_log()
            self.request_table.setRowCount(0)
            self.request_text.clear()
            self.response_text.clear()
            logging.info("Request history cleared")

    def toggle_api_key_visibility(self):
        """Toggle the visibility of the OpenAI API key input"""
        if self.openai_api_key_input.echoMode() == QLineEdit.EchoMode.Password:
            # Change to visible mode
            self.openai_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.api_key_toggle_button.setText("🔓")
            self.api_key_toggle_button.setToolTip("Hide API Key")
        else:
            # Change to password mode
            self.openai_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_key_toggle_button.setText("🔒")
            self.api_key_toggle_button.setToolTip("Show API Key")

    def handle_server_error(self, error_message):
        """Handle server errors"""
        logging.error(f"Server error: {error_message}")
        
        # Update UI to reflect server stopped state
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_field.setText("Error")
        self.status_field.setStyleSheet("font-weight: bold; color: #f44336;")
        self.status_label.setText(error_message)
        
        # If the thread is still running, stop it
        if self.server_thread and self.server_thread.running:
            self.server_thread.stop()

def main():
    # Configure application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 