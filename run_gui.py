import sys
import json
import logging
import time
import datetime
import subprocess
import socket
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QTableWidget, QTableWidgetItem, QLineEdit,
    QComboBox, QCheckBox, QGroupBox, QSplitter, QGridLayout, QHeaderView, QSpinBox, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QFont, QTextCursor, QColor

from core.router import OllamaRouter
from core.util import load_config, start_localhost_run_tunnel
import core.api as api

import uvicorn
import asyncio
import threading
import platform

class LoggerTextSignal(QObject):
    """Simple QObject with a signal to safely pass logging messages across threads"""
    signal = pyqtSignal(str, str)

class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.text_edit.setReadOnly(True)
        # Use Courier font instead of Monospace (more widely available)
        self.text_edit.setFont(QFont("Courier", 9))
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        # Add a signal to safely update the text edit from any thread
        self.signal = LoggerTextSignal()
        self.signal.signal.connect(self.update_text_edit)

    def emit(self, record):
        try:
            msg = self.formatter.format(record)
            # Use signal to safely update the UI from any thread
            self.signal.signal.emit(record.levelname, msg)
        except Exception as e:
            print(f"Error in logger emit: {str(e)}")
    
    def update_text_edit(self, level, message):
        try:
            if level == 'ERROR':
                self.text_edit.append(f'<span style="color: #f44336;">{message}</span>')
            elif level == 'WARNING':
                self.text_edit.append(f'<span style="color: #FF9800;">{message}</span>')
            elif level == 'INFO':
                self.text_edit.append(f'<span style="color: #2196F3;">{message}</span>')
            else:
                self.text_edit.append(message)
            
            # Keep the latest messages visible by moving cursor to the end
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_edit.setTextCursor(cursor)
        except Exception as e:
            print(f"Error updating text edit: {str(e)}")


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
        self.start_time = time.time()
        self.last_update_time = None
        
        # Set initial status
        if response_data:
            if "error" in response_data:
                self.status = "Error"
            else:
                self.status = "Completed"
        else:
            self.status = "Pending"

    def update_chunk_count(self, count):
        """Update the chunk count and last update time"""
        self.completed_chunks = count
        self.last_update_time = time.time()
        return self
    
    def mark_completed(self):
        """Mark the request as completed"""
        self.status = "Completed"
        self.last_update_time = time.time()
        return self
    
    def mark_error(self, error_message=None):
        """Mark the request as errored"""
        self.status = "Error"
        self.last_error = error_message
        if error_message:
            if not self.response_data:
                self.response_data = {}
            self.response_data["error"] = {"message": error_message}
        self.last_update_time = time.time()
        return self


class ServerThread(QThread):
    """Thread to run the uvicorn server"""
    update_signal = pyqtSignal(str)
    tunnel_url_signal = pyqtSignal(str)
    request_update_signal = pyqtSignal()
    
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
        class SignalHandler(logging.Handler):
            def __init__(self, signal):
                super().__init__()
                self.signal = signal
                self.formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
                
            def emit(self, record):
                msg = self.formatter.format(record)
                self.signal.emit(msg)
        
        log_handler = SignalHandler(self.update_signal)
        log_handler.setLevel(logging.INFO)
        logging.getLogger("core").addHandler(log_handler)
        
        def request_callback(data):
            try:
                data_type = data.get("type", "")
                
                if data_type == "request":
                    request_data = data.get("request", {})
                    self.update_signal.emit(f"Request: {json.dumps(request_data, indent=2)}")
                    
                    entry = RequestLogEntry(request_data)
                    
                    model = request_data.get('model', 'unknown')
                    from core.router import OllamaRouter
                    router = OllamaRouter(ollama_endpoint=self.ollama_endpoint)
                    entry.model = router.get_model_name(model)
                    
                    self.request_log.append(entry)
                    self.request_update_signal.emit()
                    
                elif data_type == "response":
                    request_data = data.get("request", {})
                    response_data = data.get("response", {})
                    
                    for entry in self.request_log:
                        if entry.request_data == request_data:
                            entry.response_data = response_data
                            entry.mark_completed()
                            response_str = json.dumps(response_data)
                            log_str = response_str[:100] + "..." if len(response_str) > 100 else response_str
                            self.update_signal.emit(f"Response: {log_str}")
                            self.request_update_signal.emit()
                            break
                
                elif data_type == "stream_start":
                    request_data = data.get("request", {})
                    
                    for entry in self.request_log:
                        if entry.request_data == request_data:
                            entry.is_streaming = True
                            self.update_signal.emit(f"Stream started for request")
                            self.request_update_signal.emit()
                            break
                
                elif data_type == "stream_chunk":
                    request_data = data.get("request", {})
                    chunk_count = data.get("chunk_count", 0)
                    
                    for entry in self.request_log:
                        if entry.request_data == request_data:
                            entry.update_chunk_count(chunk_count)
                            self.update_signal.emit(f"Received {chunk_count} chunks")
                            self.request_update_signal.emit()
                            break
                
                elif data_type == "stream_end":
                    request_data = data.get("request", {})
                    chunk_count = data.get("chunk_count", 0)
                    
                    for entry in self.request_log:
                        if entry.request_data == request_data:
                            entry.update_chunk_count(chunk_count)
                            entry.mark_completed()
                            self.update_signal.emit(f"Stream completed with {chunk_count} chunks")
                            self.request_update_signal.emit()
                            break
                
                elif data_type == "error":
                    error = data.get("error", "Unknown error")
                    self.update_signal.emit(f"Error: {error}")
                    
                    request_data = data.get("request", {})
                    if request_data:
                        for entry in self.request_log:
                            if entry.request_data == request_data:
                                entry.mark_error(error)
                                self.request_update_signal.emit()
                                break
                    else:
                        for entry in self.request_log:
                            if entry.status == "Pending":
                                entry.mark_error(error)
                                self.request_update_signal.emit()
                
            except Exception as e:
                self.update_signal.emit(f"Error processing request callback: {str(e)}")
        
        try:
            # Create API
            self.app = api.create_api(
                ollama_endpoint=self.ollama_endpoint,
                api_key=self.api_key,
                request_callback=request_callback
            )
            
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="warning"
            )
            self.server = uvicorn.Server(config)
            self.running = True
            
            if self.use_tunnel:
                # Create and store a main event loop for better cleanup
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                
                async def start_gui_tunnel():
                    try:
                        self.update_signal.emit("Starting localhost.run tunnel...")
                        
                        result = await start_localhost_run_tunnel(self.port)
                        
                        if not result:
                            self.update_signal.emit("Could not get tunnel URL within timeout period.")
                            self.update_signal.emit("Will retry in 10 seconds...")
                            await asyncio.sleep(10)
                            
                            self.update_signal.emit("Retrying tunnel connection...")
                            result = await start_localhost_run_tunnel(self.port)
                            
                            if not result:
                                self.update_signal.emit("Failed to establish tunnel after retry. Please check your network connection.")
                                return
                        
                        tunnel_url, process = result
                        
                        if tunnel_url.startswith("https://admin.localhost.run"):
                            self.update_signal.emit("Error: Received admin URL instead of tunnel URL. This will not work with Cursor AI.")
                            self.update_signal.emit("Please retry with a different method.")

                            if process and process.poll() is None:
                                try:
                                    process.terminate()
                                except:
                                    pass
                            return
                        
                        self.update_signal.emit(f"Tunnel successfully started!")
                        self.update_signal.emit(f"Tunnel URL: {tunnel_url}")
                        self.tunnel_url_signal.emit(tunnel_url)

                        watchdog_time = time.time()
                        while self.running:
                            if process.poll() is not None:
                                self.update_signal.emit("Tunnel process terminated unexpectedly")
                                if self.running:
                                    self.update_signal.emit("Attempting to restart localhost.run tunnel...")
                                    result = await start_localhost_run_tunnel(self.port)
                                    if result:
                                        new_tunnel_url, process = result
                                        
                                        if not new_tunnel_url.startswith("https://admin.localhost.run"):
                                            tunnel_url = new_tunnel_url
                                            self.tunnel_url_signal.emit(tunnel_url)
                                            self.update_signal.emit(f"Tunnel restarted successfully!")
                                            self.update_signal.emit(f"New tunnel URL: {tunnel_url}")
                                        else:
                                            self.update_signal.emit("Error: Received admin URL on restart. This will not work with Cursor AI.")
                            
                            current_time = time.time()
                            if current_time - watchdog_time > 60:
                                self.update_signal.emit("Tunnel status: Active and running")
                                watchdog_time = current_time
                                
                            await asyncio.sleep(5)
                            
                            # Exit the loop if we're no longer running
                            if not self.running:
                                break
                        
                        if process and process.poll() is None:
                            self.update_signal.emit("Stopping localhost.run tunnel...")
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
                
                try:
                    tunnel_task = self.loop.create_task(start_gui_tunnel())
                    
                    # Use run_until_complete with proper error handling
                    try:
                        self.loop.run_until_complete(self.server.serve())
                    except asyncio.CancelledError:
                        self.update_signal.emit("Server task was cancelled")
                    except Exception as e:
                        self.update_signal.emit(f"Server error: {str(e)}")
                    
                    # Ensure clean event loop shutdown
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    
                    if not self.loop.is_closed():
                        # Run the event loop until all tasks are complete or cancelled
                        if pending:
                            try:
                                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            except Exception:
                                pass
                    
                    # Clean closure of the event loop when done    
                    if not self.loop.is_closed():
                        self.loop.close()
                
                except Exception as e:
                    self.update_signal.emit(f"Error in server/tunnel loop: {str(e)}")
                    # Ensure loop is closed even on error
                    if hasattr(self, 'loop') and not self.loop.is_closed():
                        try:
                            self.loop.close()
                        except Exception as loop_err:
                            self.update_signal.emit(f"Error closing event loop: {str(loop_err)}")
            else:
                try:
                    # Create an event loop for direct server management
                    self.loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.loop)
                    
                    try:
                        self.loop.run_until_complete(self.server.serve())
                    except asyncio.CancelledError:
                        self.update_signal.emit("Server task was cancelled")
                    except Exception as e:
                        self.update_signal.emit(f"Server error: {str(e)}")
                    
                    # Ensure clean event loop shutdown
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    
                    if not self.loop.is_closed():
                        # Run the event loop until all tasks are complete or cancelled
                        if pending:
                            try:
                                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            except Exception:
                                pass
                    
                    # Clean closure of the event loop when done
                    if not self.loop.is_closed():
                        self.loop.close()
                    
                except Exception as e:
                    self.update_signal.emit(f"Error running server: {str(e)}")
                    # Ensure loop is closed even on error
                    if hasattr(self, 'loop') and not self.loop.is_closed():
                        try:
                            self.loop.close()
                        except Exception as loop_err:
                            self.update_signal.emit(f"Error closing event loop: {str(loop_err)}")
        except Exception as e:
            self.update_signal.emit(f"Server startup error: {str(e)}")
    
    def stop(self):
        """Safely stop the server thread"""
        self.running = False
        
        # Ensure server has should_exit flag set
        if hasattr(self, 'server') and self.server:
            if hasattr(self.server, "should_exit"):
                self.server.should_exit = True
            if hasattr(self.server, "force_exit"):
                self.server.force_exit = True
                
        # Cancel any pending tasks in the event loop
        if hasattr(self, 'loop') and self.loop and not self.loop.is_closed():
            try:
                # Cancel all tasks
                for task in asyncio.all_tasks(self.loop):
                    task.cancel()
            except Exception as e:
                logging.error(f"Error cancelling tasks: {str(e)}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OllamaLink Dashboard")
        self.setGeometry(100, 100, 1200, 800)
        self.server_thread = None
        self.router = None
        self.config = load_config()
        
        # Create a timer for periodic UI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_request_display)
        self.update_timer.setInterval(1000)  # Update every second
            
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
        tunnel_url_label = QLabel("<b>Tunnel URL:</b>")
        
        # Set fixed width for labels to ensure alignment
        label_width = 120
        status_label.setFixedWidth(label_width)
        port_label.setFixedWidth(label_width)
        local_url_label.setFixedWidth(label_width)
        tunnel_url_label.setFixedWidth(label_width)
        
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
        
        # Tunnel URL with copy button
        tunnel_url_layout = QHBoxLayout()
        tunnel_url_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        self.tunnel_url_field = QLabel("-")
        self.tunnel_url_field.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.tunnel_url_field.setStyleSheet("font-weight: bold;")
        
        # Copy button for Tunnel URL
        self.copy_tunnel_button = QPushButton("Copy")
        self.copy_tunnel_button.setStyleSheet("background-color: #2196F3; color: white; padding: 4px; border-radius: 4px;")
        self.copy_tunnel_button.setFixedWidth(50)
        self.copy_tunnel_button.clicked.connect(self.copy_tunnel_url)
        self.copy_tunnel_button.setEnabled(False)
        
        tunnel_url_layout.addWidget(self.tunnel_url_field, 1)  # Add stretch factor
        tunnel_url_layout.addWidget(self.copy_tunnel_button, 0)  # No stretch
        
        # Add fields to grid layout - use 4 rows x 2 columns
        status_layout.addWidget(status_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.status_field, 0, 1, Qt.AlignmentFlag.AlignLeft)
        
        status_layout.addWidget(port_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addWidget(self.port_field, 1, 1, Qt.AlignmentFlag.AlignLeft)
        
        status_layout.addWidget(local_url_label, 2, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addLayout(local_url_layout, 2, 1)
        
        status_layout.addWidget(tunnel_url_label, 3, 0, Qt.AlignmentFlag.AlignLeft)
        status_layout.addLayout(tunnel_url_layout, 3, 1)
        
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
        self.request_table.setAlternatingRowColors(True)
        self.request_table.setStyleSheet("alternate-background-color: #f2f2f2;")
        self.request_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.request_table.itemClicked.connect(self.on_request_selected)
        self.request_table.setMinimumHeight(200)  # Set minimum height for request table
        
        # Set column width distribution
        header = self.request_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  # Timestamp
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)           # Model
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  # Messages
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)  # Status
        
        # Make sure the table takes the full width of its container
        self.request_table.horizontalHeader().setStretchLastSection(False) 
        
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
        
        self.tunnel_checkbox = QCheckBox("Use localhost.run Tunnel")
        self.tunnel_checkbox.setChecked(self.config["tunnel"]["use_tunnel"])
        
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
        
        # Add thinking mode toggle
        thinking_label = QLabel("Thinking Mode:")
        thinking_label.setFixedWidth(label_width)
        self.thinking_checkbox = QCheckBox("Enable thinking mode")
        # Set initial state from config
        thinking_enabled = True  # Default value
        if "ollama" in self.config and "thinking_mode" in self.config["ollama"]:
            thinking_enabled = self.config["ollama"]["thinking_mode"]
        self.thinking_checkbox.setChecked(thinking_enabled)
        thinking_help = QLabel("When disabled, /no_think prefix will be automatically added to all prompts")
        thinking_help.setStyleSheet("color: #666; font-style: italic;")
        
        # Add tooltip explaining the feature in more detail
        tooltip_text = """
        Thinking Mode controls how models process requests:
        
        ENABLED: Normal operation with full "thinking" process
        DISABLED: Adds /no_think prefix to prompt the model to respond more directly
        
        Disabling thinking mode makes responses faster but potentially less thorough.
        """
        self.thinking_checkbox.setToolTip(tooltip_text)
        thinking_label.setToolTip(tooltip_text)
        
        # Add integrity check toggle
        integrity_label = QLabel("Integrity Checks:")
        integrity_label.setFixedWidth(label_width)
        self.skip_integrity_checkbox = QCheckBox("Skip model integrity checks")
        # Set initial state from config
        skip_integrity = False  # Default value
        if "ollama" in self.config and "skip_integrity_check" in self.config["ollama"]:
            skip_integrity = self.config["ollama"]["skip_integrity_check"]
        self.skip_integrity_checkbox.setChecked(skip_integrity)
        integrity_help = QLabel("Enable to skip model integrity checks if you experience timeouts")
        integrity_help.setStyleSheet("color: #666; font-style: italic;")
        
        # Add tooltip explaining the feature in more detail
        integrity_tooltip = """
        Model integrity checks help detect corrupt model files early,
        but can cause timeout errors with large models or slow systems.
        
        If you see 'Read timed out' errors in the logs, try enabling this option.
        """
        self.skip_integrity_checkbox.setToolTip(integrity_tooltip)
        integrity_label.setToolTip(integrity_tooltip)
        
        # Add token limit control
        token_label = QLabel("Max Tokens:")
        token_label.setFixedWidth(label_width)
        self.token_limit_spinner = QSpinBox()
        self.token_limit_spinner.setMinimum(2000)
        self.token_limit_spinner.setMaximum(32000)
        self.token_limit_spinner.setSingleStep(1000)
        # Default to 32000 if not in config
        token_limit = 32000
        if "ollama" in self.config and "max_streaming_tokens" in self.config["ollama"]:
            token_limit = self.config["ollama"]["max_streaming_tokens"]
        self.token_limit_spinner.setValue(token_limit)
        token_help = QLabel("Higher values keep more context but may use more memory")
        token_help.setStyleSheet("color: #666; font-style: italic;")
        
        # Add to layout
        row = 0
        
        settings_layout.addWidget(host_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.host_input, row, 1)
        row += 1
        
        settings_layout.addWidget(port_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.port_input, row, 1)
        row += 1
        
        settings_layout.addWidget(tunnel_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.tunnel_checkbox, row, 1)
        row += 1
        
        settings_layout.addWidget(ollama_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.ollama_endpoint_input, row, 1)
        row += 1
        
        settings_layout.addWidget(openai_key_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addLayout(api_key_layout, row, 1)
        row += 1
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #ddd;")
        settings_layout.addWidget(separator, row, 0, 1, 2)
        row += 1
        
        # Add thinking mode
        settings_layout.addWidget(thinking_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.thinking_checkbox, row, 1)
        row += 1
        settings_layout.addWidget(QLabel(), row, 0)  # Empty spacer
        settings_layout.addWidget(thinking_help, row, 1)
        row += 1
        
        # Add integrity check
        settings_layout.addWidget(integrity_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.skip_integrity_checkbox, row, 1)
        row += 1
        settings_layout.addWidget(QLabel(), row, 0)  # Empty spacer
        settings_layout.addWidget(integrity_help, row, 1)
        row += 1
        
        # Add token limit
        settings_layout.addWidget(token_label, row, 0, Qt.AlignmentFlag.AlignLeft)
        settings_layout.addWidget(self.token_limit_spinner, row, 1)
        row += 1
        settings_layout.addWidget(QLabel(), row, 0)  # Empty spacer
        settings_layout.addWidget(token_help, row, 1)
        row += 1
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        save_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px; font-weight: bold; border-radius: 4px;")
        save_button.setMinimumWidth(120)
        
        # Add spacer and button at bottom
        settings_layout.addWidget(QWidget(), row, 0, 1, 2)  # Empty spacer
        settings_layout.setRowStretch(row, 1)  # Make the spacer expand
        row += 1
        settings_layout.addWidget(save_button, row, 0, 1, 2, Qt.AlignmentFlag.AlignCenter)
        
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
        
        # Save thinking mode setting
        self.config["ollama"]["thinking_mode"] = self.thinking_checkbox.isChecked()
        
        # Save integrity check setting
        self.config["ollama"]["skip_integrity_check"] = self.skip_integrity_checkbox.isChecked()
        
        # Save token limit setting
        self.config["ollama"]["max_streaming_tokens"] = self.token_limit_spinner.value()
        
        # Save to file
        try:
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Model mappings saved to config.json: {json.dumps(mappings)}")
            logging.info(f"Thinking mode saved: {self.thinking_checkbox.isChecked()}")
            logging.info(f"Skip integrity checks: {self.skip_integrity_checkbox.isChecked()}")
            logging.info(f"Max streaming tokens: {self.token_limit_spinner.value()}")
            
            # Reinitialize router with new settings
            self.init_router()
            self.update_model_display()
        except Exception as e:
            logging.error(f"Error saving model mappings: {str(e)}")

    def save_settings(self):
        """Save settings to config.json"""
        self.config["server"]["hostname"] = self.host_input.text()
        self.config["server"]["port"] = int(self.port_input.text())
        self.config["tunnel"]["use_tunnel"] = self.tunnel_checkbox.isChecked()
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
            
            # Display thinking mode status
            thinking_status = "Enabled" if self.router.thinking_mode else "Disabled (using /no_think)"
            self.available_models_text.append(f"<b>Thinking mode:</b> {thinking_status}\n")
            
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
    
    def copy_tunnel_url(self):
        """Copy Tunnel URL to clipboard"""
        url = self.tunnel_url_field.text()
        if url and url != "-" and url != "Disabled" and url != "Starting...":
            clipboard = QApplication.clipboard()
            clipboard.setText(url)
            logging.info(f"Copied to clipboard: {url}")
            
            # Show visual feedback
            original_text = self.copy_tunnel_button.text()
            original_style = self.copy_tunnel_button.styleSheet()
            
            self.copy_tunnel_button.setText("✓")
            self.copy_tunnel_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 4px;")
            
            # Reset after a short delay
            def reset_button():
                self.copy_tunnel_button.setText(original_text)
                self.copy_tunnel_button.setStyleSheet(original_style)
            
            QTimer.singleShot(1500, reset_button)

    def update_tunnel_url(self, url):
        """Update the tunnel URL"""
        if not url:
            self.tunnel_url_field.setText("No tunnel URL available")
            self.copy_tunnel_button.setEnabled(False)
            logging.error("Received empty tunnel URL")
            return
            
        # Check for admin.localhost.run which should be rejected
        if url.startswith("https://admin.localhost.run"):
            self.tunnel_url_field.setText("❌ ERROR: admin.localhost.run received (won't work!)")
            self.copy_tunnel_button.setEnabled(False)
            self.tunnel_url_field.setStyleSheet("color: red; font-weight: bold;")
            logging.error(f"CRITICAL ERROR: Received admin URL '{url}'. This URL will NOT work with Cursor AI.")
            logging.error("You must restart with a proper tunnel that doesn't use the admin URL.")
            return
            
        # Valid tunnel URL - format as API URL
        api_url = f"{url}/v1"
        self.tunnel_url_field.setText(api_url)
        self.copy_tunnel_button.setEnabled(True)
        self.tunnel_url_field.setStyleSheet("color: green; font-weight: bold;")
        
        # Log with domain information for clarity
        if ".lhr.life" in url:
            logging.info(f"Tunnel URL set (lhr.life domain): {api_url}")
        else:
            logging.info(f"Tunnel URL set: {api_url}")
        
        # Update the status bar too
        local_url = self.url_field.text()
        self.status_label.setText(f"Running locally on {local_url} and via tunnel at {api_url}")
    
    def is_port_in_use(self, port):
        """Check if a port is already in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except OSError:
                return True
    
    def find_available_port(self, start_port, max_attempts=10):
        """Find an available port starting from start_port"""
        current_port = start_port
        attempts = 0
        
        while attempts < max_attempts:
            if not self.is_port_in_use(current_port):
                return current_port
            current_port += 1
            attempts += 1
        
        # If we couldn't find an available port, return the original port
        # and let the server handle the error
        return start_port
    
    def start_server(self):
        """Start the FastAPI server"""
        if self.server_thread and self.server_thread.running:
            return
        
        # Disable start button immediately to prevent double clicks
        self.start_button.setEnabled(False)
        
        # Get server configuration
        try:
            host = self.host_input.text().strip()
            if not host:
                host = "127.0.0.1"  
                self.host_input.setText(host)
            
            try:
                port = int(self.port_input.text())
                if port <= 0 or port > 65535:
                    raise ValueError("Port must be between 1 and 65535")
            except ValueError as e:
                logging.error(f"Invalid port number: {str(e)}")
                port = 8000 
                self.port_input.setText(str(port))
            if self.is_port_in_use(port):
                original_port = port
                port = self.find_available_port(port)
                
                if port != original_port:
                    logging.warning(f"Port {original_port} is already in use. Using port {port} instead.")

                    self.config["server"]["port"] = port
                    self.port_input.setText(str(port))
                else:
                    logging.error(f"Port {original_port} is already in use and no alternative ports are available.")

                    self.start_button.setEnabled(True)
                    return
            
            ollama_endpoint = self.ollama_endpoint_input.text().strip()
            if not ollama_endpoint:
                ollama_endpoint = "http://localhost:11434" 
                self.ollama_endpoint_input.setText(ollama_endpoint)
                
            use_tunnel = self.tunnel_checkbox.isChecked()
            api_key = None
            if "openai" in self.config and "api_key" in self.config["openai"]:
                api_key = self.config["openai"]["api_key"]
            
            if self.server_thread:
                try:
                    if self.server_thread.isRunning():
                        self.server_thread.running = False
                        self.server_thread.terminate()
                    self.server_thread = None
                except Exception as e:
                    logging.error(f"Error cleaning up old server thread: {str(e)}")
                    self.server_thread = None  

            self.server_thread = ServerThread(
                port=port,
                host=host,
                ollama_endpoint=ollama_endpoint,
                use_tunnel=use_tunnel,
                router=self.router,
                api_key=api_key
            )
            
            self.server_thread.update_signal.connect(self.update_request_log)
            self.server_thread.tunnel_url_signal.connect(self.update_tunnel_url)
            self.server_thread.request_update_signal.connect(self.refresh_request_display)
            
            # Start the server
            self.server_thread.start()
            
            # Start the update timer
            self.update_timer.start()
            
            # Update UI
            self.stop_button.setEnabled(True)
            self.status_field.setText("Running")
            self.status_field.setStyleSheet("font-weight: bold; color: #4CAF50;")
            self.port_field.setText(str(port))
            base_url = f"http://{host}:{port}/v1"
            self.url_field.setText(base_url)
            self.status_label.setText(f"Running on {base_url}")
            
            # Reset Tunnel URL if not using tunnel
            if not use_tunnel:
                self.tunnel_url_field.setText("Disabled")
                self.copy_tunnel_button.setEnabled(False)
            else:
                self.tunnel_url_field.setText("Starting...")
                self.copy_tunnel_button.setEnabled(False)

            self.update_model_display()
            
            logging.info(f"Server started on {base_url}")
            
        except Exception as e:
            logging.error(f"Error starting server: {str(e)}")
            self.start_button.setEnabled(True)
            if self.server_thread:
                try:
                    self.server_thread.terminate()
                except:
                    pass
                self.server_thread = None
    
    def stop_server(self):
        """Stop the server"""
        if self.server_thread and self.server_thread.running:
            logging.info("Stopping server...")
            self.status_field.setText("Stopping...")
            self.status_field.setStyleSheet("font-weight: bold; color: #FFA000;")
            self.update_timer.stop()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.server_thread.running = False
            

            try:
                if hasattr(self.server_thread, 'server') and self.server_thread.server:
                    def force_shutdown():
                        try:
                            if hasattr(self.server_thread.server, "should_exit"):
                                self.server_thread.server.should_exit = True
                            if hasattr(self.server_thread.server, "force_exit"):
                                self.server_thread.server.force_exit = True
                            # Explicitly handle event loop closure for Lifespan
                            if hasattr(self.server_thread, 'app') and self.server_thread.app:
                                # Need to properly cleanup lifespan tasks
                                if hasattr(self.server_thread.server, "lifespan"):
                                    lifespan = self.server_thread.server.lifespan
                                    if hasattr(lifespan, "shutdown_event"):
                                        lifespan.shutdown_event.set()
                                    if hasattr(lifespan, "receive_queue"):
                                        # Clear any pending queue items to avoid blocking
                                        try:
                                            if not lifespan.receive_queue.empty():
                                                while not lifespan.receive_queue.empty():
                                                    lifespan.receive_queue.get_nowait()
                                        except Exception:
                                            pass
                        except Exception as e:
                            logging.error(f"Error in force_shutdown: {str(e)}")
                    
                    threading.Thread(target=force_shutdown, daemon=True).start()
            except Exception as e:
                logging.error(f"Error forcing server shutdown: {str(e)}")
            
            max_wait_time = 5000  # 5 seconds
            check_interval = 100  # 100ms
            checks_performed = 0
            max_checks = max_wait_time // check_interval
            
            def check_thread_stopped():
                nonlocal checks_performed
                checks_performed += 1
                
                # Check if the thread was terminated already
                if self.server_thread is None:
                    self.update_ui_after_stop()
                    return
                
                if not self.server_thread.isRunning():
                    # Create a temporary reference to avoid access after deletion
                    temp_thread = self.server_thread
                    self.server_thread = None
                    try:
                        # Clean up thread resources (if any)
                        del temp_thread
                    except:
                        pass
                    
                    self.update_ui_after_stop()
                    return
                
                if checks_performed >= max_checks:
                    logging.warning("Server thread didn't stop within timeout, forcing termination")
                    try:
                        # Terminate the thread safely
                        temp_thread = self.server_thread
                        self.server_thread = None
                        temp_thread.terminate()
                        
                        # Try to clean up resources
                        del temp_thread
                    except Exception as e:
                        logging.error(f"Error terminating thread: {str(e)}")
                    
                    self.update_ui_after_stop()
                    
                    # Kill any processes that might be blocking the port
                    try:
                        port = self.config["server"]["port"]
                        platform_system = platform.system()
                        
                        if platform_system == "Windows":
                            subprocess.run(f"FOR /F \"tokens=5\" %a in ('netstat -ano ^| findstr :{port}') do taskkill /F /PID %a", shell=True)
                        elif platform_system == "Darwin":  # macOS
                            subprocess.run(f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}' | xargs -r kill -9 2>/dev/null || true", shell=True)
                        else:  # Linux and others
                            subprocess.run(f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}' | xargs -r kill -9", shell=True)
                        
                        logging.info(f"Forcibly released port {port}")
                    except Exception as e:
                        logging.error(f"Error cleaning up port: {str(e)}")
                    
                    return
                
                QTimer.singleShot(check_interval, check_thread_stopped)
            
            QTimer.singleShot(check_interval, check_thread_stopped)
            
        else:
            self.update_ui_after_stop()

    def update_ui_after_stop(self):
        """Update UI elements after server has stopped"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_field.setText("Not Running")
        self.status_field.setStyleSheet("font-weight: bold; color: #f44336;")
        self.status_label.setText("Not Running")
        self.tunnel_url_field.setText("-")
        self.copy_tunnel_button.setEnabled(False)
        
        self.update_timer.stop()
        
        logging.info("Server stopped")
    
    def update_request_log(self, message):
        """Update the request log with a new message"""
        logging.info(message)
    
    def refresh_request_display(self):
        """Refresh the request table display with current data"""
        if self.server_thread and hasattr(self.server_thread, "request_log"):
            current_row = self.request_table.currentRow()
            
            self.request_table.setRowCount(0)
            
            for i, entry in enumerate(self.server_thread.request_log):
                self.request_table.insertRow(i)
                
                dt = datetime.datetime.fromtimestamp(entry.timestamp)
                time_str = dt.strftime("%H:%M:%S")
                
                self.request_table.setItem(i, 0, QTableWidgetItem(time_str))
                self.request_table.setItem(i, 1, QTableWidgetItem(entry.model))
                
                msg_count = len(entry.messages)
                self.request_table.setItem(i, 2, QTableWidgetItem(f"{msg_count} messages"))
                
                status_text = entry.status
                
                current_time = time.time()
                is_stale = False
                
                if hasattr(entry, 'last_update_time') and entry.last_update_time:
                    time_since_update = current_time - entry.last_update_time
                    if time_since_update > 30 and entry.status == "Pending":
                        is_stale = True
                elif hasattr(entry, 'start_time'):
                    time_since_start = current_time - entry.start_time
                    if time_since_start > 60 and entry.status == "Pending":
                        is_stale = True
                
                if is_stale:
                    entry.mark_completed()
                    status_text = "Completed"
                
                if entry.is_streaming and entry.status == "Pending" and entry.completed_chunks > 0:
                    status_text = f"Streaming ({entry.completed_chunks})"
                elif entry.is_streaming and entry.status == "Completed" and entry.completed_chunks > 0:
                    status_text = f"Completed ({entry.completed_chunks})"
                
                status_item = QTableWidgetItem(status_text)
                
                if entry.status == "Completed":
                    status_item.setForeground(QColor("#4CAF50"))  
                elif entry.status == "Timeout":
                    status_item.setForeground(QColor("#FF9800"))  
                elif entry.status == "Error":
                    status_item.setForeground(QColor("#F44336")) 
                elif status_text.startswith("Streaming"):
                    status_item.setForeground(QColor("#2196F3"))  
                
                self.request_table.setItem(i, 3, status_item)
            
            header = self.request_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)  
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)          
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)  
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents) 
            
            if (current_row < 0 or current_row >= self.request_table.rowCount()) and self.request_table.rowCount() > 0:
                select_row = min(max(0, current_row), self.request_table.rowCount() - 1)
                self.request_table.selectRow(select_row)
                self.on_request_selected(self.request_table.item(select_row, 0))
            elif current_row >= 0 and current_row < self.request_table.rowCount():
                self.request_table.selectRow(current_row)

    def on_request_selected(self, item):
        """Handle request selection from the request table"""
        row = item.row()
        
        if self.server_thread and hasattr(self.server_thread, "request_log"):
            if row < len(self.server_thread.request_log):
                entry = self.server_thread.request_log[row]
                
                try:
                    formatted_request = json.dumps(entry.request_data, indent=2)
                    self.request_text.setText(formatted_request)
                except Exception as e:
                    self.request_text.setText(f"Error formatting request: {str(e)}")
                
                if entry.response_data:
                    try:
                        formatted_response = json.dumps(entry.response_data, indent=2)
                        
                        if entry.status == "Error" and "error" in entry.response_data:
                            error_msg = entry.response_data["error"].get("message", "Unknown error")
                            self.response_text.setHtml(f'<span style="color: #F44336; font-weight: bold;">ERROR: {error_msg}</span><br><br>' + 
                                                    f'<pre>{formatted_response}</pre>')
                        else:
                            self.response_text.setText(formatted_response)
                    except Exception as e:
                        self.response_text.setText(f"Error formatting response: {str(e)}")
                else:
                    if entry.status == "Timeout":
                        self.response_text.setHtml('<span style="color: #FF9800; font-weight: bold;">Request timed out</span>')
                    elif entry.status == "Error":
                        if hasattr(entry, 'last_error') and entry.last_error:
                            self.response_text.setHtml(f'<span style="color: #F44336; font-weight: bold;">Error: {entry.last_error}</span>')
                        else:
                            self.response_text.setHtml('<span style="color: #F44336; font-weight: bold;">Error occurred</span>')
                    elif entry.is_streaming:
                        elapsed = ""
                        if hasattr(entry, 'start_time'):
                            if hasattr(entry, 'last_update_time') and entry.last_update_time:
                                elapsed = f" in {entry.last_update_time - entry.start_time:.1f}s"
                            elif entry.status == "Completed":
                                elapsed = f" in {time.time() - entry.start_time:.1f}s"
                        
                        if entry.status == "Completed":
                            self.response_text.setHtml(
                                '<span style="color: #4CAF50; font-weight: bold;">Streaming request completed</span><br>' +
                                f'<span>Received {entry.completed_chunks} chunks{elapsed}</span>'
                            )
                        elif entry.completed_chunks > 0:
                            self.response_text.setHtml(
                                '<span style="color: #2196F3; font-weight: bold;">Streaming in progress</span><br>' +
                                f'<span>Received {entry.completed_chunks} chunks so far...</span>'
                            )
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
            self.openai_api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.api_key_toggle_button.setText("🔓")
            self.api_key_toggle_button.setToolTip("Hide API Key")
        else:
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