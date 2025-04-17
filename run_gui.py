import sys
import json
import logging
import time
import datetime
import base64
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
import core.api as api

import uvicorn
from fastapi import Request, Response
from starlette.requests import ClientDisconnect
import asyncio

# Set up logging
class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Monospace", 9))
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

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


class ServerThread(QThread):
    """Thread to run the uvicorn server"""
    update_signal = pyqtSignal(str)
    tunnel_url_signal = pyqtSignal(str)
    
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
        self.request_log = []
        
    def run(self):

        # Create the custom app
        self.app = api.create_api(ollama_endpoint=self.ollama_endpoint, api_key=self.api_key)
        
        # Add middleware to capture requests/responses
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            # Log the request
            try:
                # Clone the request body to avoid consuming it
                body_bytes = await request.body()
                request_data = {}
                
                # Try to parse as JSON
                try:
                    request_data = json.loads(body_bytes.decode())
                except json.JSONDecodeError:
                    # Not JSON or empty body
                    request_data = {"non_json_body": True}
                
                # Create a new request with the same body
                async def receive():
                    return {"type": "http.request", "body": body_bytes, "more_body": False}
                
                # Replace the receive method to return the stored body
                request._receive = receive
                
                # Extract the path for endpoint detection
                path = request.url.path
                is_chat_completion = "/v1/chat/completions" in path
                
                # Store request
                if request_data:
                    self.update_signal.emit(f"Request: {json.dumps(request_data, indent=2)}")
                    entry = RequestLogEntry(request_data)
                    
                    # Better model detection
                    if is_chat_completion and (not entry.model or entry.model == "unknown"):
                        # Try to identify model from the router if available
                        if hasattr(self, 'router') and self.router:
                            if "model" in request_data:
                                # Use the router to resolve the model name
                                resolved_model = self.router.get_model_name(request_data["model"])
                                entry.model = resolved_model
                            else:
                                # No model specified, use default model
                                entry.model = self.router.default_model or "unknown"
                    
                    self.request_log.append(entry)
                
                # Process request (with timeout)
                try:
                    response = await asyncio.wait_for(call_next(request), timeout=120)  # 2 minute timeout
                    
                    # Check if this is a streaming response
                    is_streaming = False
                    if request_data and is_chat_completion:
                        is_streaming = request_data.get("stream", False)
                    
                    if is_streaming:
                        # For streaming responses, we need to process each chunk
                        # and mark the request as completed after all chunks
                        chunks = []
                        entry_to_update = None
                        
                        # Find the request entry
                        for entry in self.request_log:
                            if entry.request_data == request_data:
                                entry_to_update = entry
                                break
                        
                        # Create a modified iterator to capture chunks and update status
                        async def capture_chunks():
                            nonlocal entry_to_update
                            chunk_count = 0
                            
                            async for chunk in response.body_iterator:
                                chunks.append(chunk)
                                chunk_count += 1
                                
                                # Update the entry with progress
                                if entry_to_update:
                                    entry_to_update.completed_chunks = chunk_count
                                    
                                    # Try to parse as JSON to detect final chunk
                                    try:
                                        chunk_data = json.loads(chunk.decode())
                                        if chunk_data.get("choices", [{}])[0].get("finish_reason") is not None:
                                            entry_to_update.status = "Completed"
                                    except:
                                        pass
                                
                                yield chunk
                            
                            # Stream completed - ensure the status is updated
                            if entry_to_update:
                                entry_to_update.status = "Completed"
                                self.update_signal.emit(f"Streaming request completed with {chunk_count} chunks")
                        
                        # Return a modified response with our chunk capturing iterator
                        return Response(
                            content=capture_chunks(),
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type,
                            background=response.background
                        )
                    else:
                        # Non-streaming response - process as before
                        response_body = b""
                        async for chunk in response.body_iterator:
                            response_body += chunk
                        
                        # Try to parse as JSON for completions
                        if is_chat_completion:
                            try:
                                response_data = json.loads(response_body.decode())
                                
                                # Update request entry with response
                                if request_data:
                                    for entry in self.request_log:
                                        if entry.request_data == request_data:
                                            entry.response_data = response_data
                                            entry.status = "Completed"
                                            break
                                
                                self.update_signal.emit(f"Response: {json.dumps(response_data, indent=2)}")
                            except json.JSONDecodeError:
                                self.update_signal.emit(f"Response: {response_body.decode()}")
                        
                        # Return the response with the body we already consumed
                        return Response(
                            content=response_body,
                            status_code=response.status_code,
                            headers=dict(response.headers),
                            media_type=response.media_type
                        )
                    
                except asyncio.TimeoutError:
                    # Request took too long
                    self.update_signal.emit("Request timed out (taking longer than 2 minutes)")
                    
                    # Mark the request as timed out
                    if request_data:
                        for entry in self.request_log:
                            if entry.request_data == request_data:
                                entry.status = "Timeout"
                                break
                    
                    return Response(
                        content=json.dumps({
                            "error": {
                                "message": "Request timed out. The model is taking too long to respond.",
                                "type": "timeout",
                                "code": 504
                            }
                        }),
                        status_code=504,
                        media_type="application/json"
                    )
                    
            except ClientDisconnect:
                # Client disconnected, just log it
                self.update_signal.emit("Client disconnected before response was complete")
                return Response(
                    content=json.dumps({
                        "error": {
                            "message": "Client disconnected",
                            "type": "client_disconnect",
                            "code": 499
                        }
                    }),
                    status_code=499,
                    media_type="application/json"
                )
                
            except Exception as e:
                # Log the error
                self.update_signal.emit(f"Error capturing request/response: {str(e)}")
                return await call_next(request)
        
        # Add exception handler for client disconnects
        @self.app.exception_handler(ClientDisconnect)
        async def client_disconnect_handler(request, exc):
            self.update_signal.emit("Client disconnected")
            return Response(
                content=json.dumps({
                    "error": {
                        "message": "Client disconnected",
                        "type": "client_disconnect",
                        "code": 499
                    }
                }),
                status_code=499,
                media_type="application/json"
            )
        
        # Configure and start the server
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.create_task(start_gui_cloudflared_tunnel())
            loop.run_until_complete(self.server.serve())
        else:
            # Just run the server without tunnel
            asyncio.run(self.server.serve())
    
    def stop(self):
        """Safely stop the server thread"""
        self.running = False
        # The actual server will be stopped in the run method
        # This just signals that we should stop


# Simple Ollama icon (base64 encoded for portability)
OLLAMA_ICON = """
iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOk
AAABL1BMVEX///8AAADMzMz4+PgiIiLb29tERET09PSCgoLGxsYUFBTi4uIJCQlSUlKysrI5OTnx8fHp6emio
qIrKytlZWW5ubk9PT1bW1uOjo4TnZ0tmZnIjY0hnp4ckJBjjY0bgoIlk5MbiorChob3i4vdl5f97u7DKCjadH
Talpa0TEzrKyvrFhbkWVnwPT3LNzfAaWnGXl7kj4/Qvr7O6uru7uiVFSQYmKKV1dXnJxRlJRHl5dAjIxJhYV
1jY1qiIhBh4d6hYWzLS1jiIhKeHhFdnZfg4NbfHxegoJZfn5Td3dPhYVMf39Lc3NHeXlBcXFUdHRRc3NCe3s+
dXU8cnI4cHBFdHQ6bm42a2sxaGg0ZmYvZGQuYWE1Z2cyY2PUbW3x9/euoKCZqKil0NA8MXYPAAAEaklEQVR4Ae
3a2VoaMRzH8SyMy3RPLwQVWSJCcStqQEGtVkGt1qXWFVv3+78DJhmTWc7yfyYzzpzvU3j5HMIwmYnKBnQ1vd/o9
fRG002rbMBE48u7X+qP9Udv7y562rJsQEvLe78/0vT1Z90+MzuhsgFTe+V72/PZnnJlZQ209VeGVlbdVLdvtB/l
oobayqrmHo7qpdyO3j9e+77q9ZCWU71TPshprKP/G5VyGPH0XlvVrjUO9VGO9fXOhVXvWrOvj3P8zbqXy1jTgpn
esda3ZF7Tg29OC/X+g/BZfz5o9wutc9LQp9FY/yvx4xdD1oJXPYfnrDFb9jnpHlkj1K4dDpYDxhbK4sNvg0OQRb
HRb4BX0YbW3B9P4HUU3QKHeNtFewPqDqwRYgN4HZsX+NQbGNM1gW27pjkaW4CXWQM+A16D0O2B12HNGuDFr4a7g
LcfvQJTgGZQsyb4bBrKtQAva0wPiGrB65RKtYAJa9Y0dB0V3QcYs2ZNDaD6FYCxQM2aGlDdX9AYjKlW1KypASwC
WMaYGuP6mLKmB5QBb1S/aNDaEQAXsaZnTQvYArxWEGdtC7SwYs0KuAS8izjO2iUo1qSsaQFNwBvEcdaaoLAixZo
W0AO8QRxnrQeKNSlrWkAf8AZxnLU+KKxIsXYCkOc/dxLHWTsBxZoUa1rAmTHAWRxn7QwUa1Ks6QAOgDt4gTM71g
FAikVNiTUd4BrADTzbjjUOKNasWFujBnwGngfPsmKNA4o1OdakgCsA19Ct9K1YcwHFmiRrUsBZB7oOdEFZwTUr1
jqgWJNkTQroA+gDcGAFs2LNAcWaJGtSwOUlgEt4QRaZswIKK5KsnQJo9wHcw+uwYoxZATcr1qRZI/8HdO/h9eF54
LVuUzJrFI8BeNYFLEMz1SkZQHkOhM65gLdNLdC6qpTFWu1K0+1nfuNlLtB5rpmrXZfMWu1aM53d31YmZnzdWcOsN
WpnRNZqZ5pu7+7+/r6bWUy9xowtZhfIDNYYA0yq5IzRd6G+3mWw1qhbR2at3tB0zs/Pb87POvQqaxfIZNb0CzxQC
WWy5jlgN0MBeqMu2BoOAIV/7lZK9qwBYDO+BrxhRMGRuQZwZnY/4J2rNzJnDcDVX9EbhuMUH5izBuB4ZYq1OlWYM
WdtZbhuKwVrtUIKdqxZLxmLG8BqlZa9Bde6dADrXVLS98LrfXrqTVBa+tlIeyzIw54Pt0eCQ/S9weZI8BNGPwO5G
AlWYFQQHgrWYVZwNBSswrReHgqaEWaFfSgogWhVnguqEXYVx6HgMsK04jwUXETYVgXnQfcLUaxU4C9ZrRuI4n4C/
8/lQxRvB+5jYoQNPsG9fgZRTL2B9/CiRD9kXU8vINrPM/A+fkY0g/9eZRjQHXaBD9oF6IYdoIPGALrZB2hpJUCLt
RGg9c8F6MV7BfRqXQK9fXOAvl9LgPbIGaA2OAK0ykWA9tktQEdfAjT9PxCwc1GGAQeijv8BeJtQA17UPu8AAAAA
SUVORK5CYII=
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OllamaLink Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.server_thread = None
        self.router = None
        self.config = load_config()
        
        # Set application icon
        self.set_app_icon()
        
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
    
    def set_app_icon(self):
        """Set the application icon"""
        try:
            # Decode the base64 icon
            icon_data = base64.b64decode(OLLAMA_ICON)
            pixmap = QPixmap()
            pixmap.loadFromData(icon_data)
            icon = QIcon(pixmap)
            self.setWindowIcon(icon)
        except Exception as e:
            logging.warning(f"Failed to set application icon: {str(e)}")
    
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
        self.available_models_text.setStyleSheet("font-family: monospace; border: 1px solid #ddd; border-radius: 4px;")
        
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
        self.log_text.setStyleSheet("font-family: monospace; background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 4px; padding: 8px;")
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
        logging.getLogger().addHandler(log_handler)
    
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
        
        requests_header = QLabel("Request History")
        requests_header.setStyleSheet("font-weight: bold;")
        requests_layout.addWidget(requests_header)
        
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
        self.request_text.setStyleSheet("font-family: monospace; background-color: #f8f8f8;")
        
        request_group_layout.addWidget(self.request_text)
        
        # Response section
        response_group = QGroupBox("Response Details")
        response_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        response_group_layout = QVBoxLayout()
        response_group.setLayout(response_group_layout)
        
        self.response_text = QTextEdit()
        self.response_text.setReadOnly(True)
        self.response_text.setStyleSheet("font-family: monospace; background-color: #f8f8f8;")
        
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
        
        # Clear button at bottom
        buttons_layout = QHBoxLayout()
        
        clear_button = QPushButton("Clear History")
        clear_button.clicked.connect(self.clear_request_history)
        clear_button.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        clear_button.setMinimumWidth(120)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(clear_button)
        
        layout.addLayout(buttons_layout)
    
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
    
    def update_request_log(self, message):
        """Update the request log with a new message"""
        logging.info(message)
        
        # Update the request table if we have a server thread
        if self.server_thread and hasattr(self.server_thread, "request_log"):
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
            self.server_thread.request_log = []
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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 