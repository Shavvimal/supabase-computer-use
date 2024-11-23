import sys
import os
import platform
import json
import base64
from datetime import datetime
from functools import partial
from typing import cast
from pathlib import Path
import logging

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QSlider, QScrollArea, QGroupBox, QTextEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage

from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock

from screeninfo import get_monitors
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

from scu.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    sampling_loop_sync,
)

from scu.tools import ToolResult
from scu.tools.computer import get_screen_details

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

SELECTED_SCREEN_INDEX = None
SCREEN_NAMES = None

class Sender:
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"

def setup_state(state):
    logging.debug("Setting up state")
    if "messages" not in state:
        state["messages"] = []
    if "api_key" not in state:
        state["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
        if not state["api_key"]:
            print("API key not found. Please set it in the .env file.")
        else:
            print("API key successfully loaded.")
    if "provider" not in state:
        state["provider"] = os.getenv("API_PROVIDER", "anthropic") or APIProvider.ANTHROPIC
    if "provider_radio" not in state:
        state["provider_radio"] = state["provider"]
    if "model" not in state:
        _reset_model(state)
    if "auth_validated" not in state:
        state["auth_validated"] = False
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2  # 10
    if "custom_system_prompt" not in state:
        state["custom_system_prompt"] = ""
        device_os_name = (
            "Windows"
            if platform.system() == "Windows"
            else "Mac"
            if platform.system() == "Darwin"
            else "Linux"
        )
        state["custom_system_prompt"] += f"\n\nNOTE: you are operating a {device_os_name} machine"
    if "hide_images" not in state:
        state["hide_images"] = False

def _reset_model(state):
    state["model"] = PROVIDER_TO_DEFAULT_MODEL_NAME[cast(APIProvider, state["provider"])]

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def _render_message(sender: str, message, state):
    is_tool_result = not isinstance(message, str) and (
        isinstance(message, ToolResult)
        or message.__class__.__name__ == "ToolResult"
        or message.__class__.__name__ == "CLIResult"
    )
    if not message or (
        is_tool_result
        and state["hide_images"]
        and not hasattr(message, "error")
        and not hasattr(message, "output")
    ):
        return
    if is_tool_result:
        message = cast(ToolResult, message)
        if message.output:
            return message.output
        if message.error:
            return f"Error: {message.error}"
        if message.base64_image and not state["hide_images"]:
            return base64.b64decode(message.base64_image)
    elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
        return message.text
    elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
        return f"Tool Use: {message.name}\nInput: {message.input}"
    else:
        return message

def accumulate_messages(*args, **kwargs):
    accumulated_messages = []
    global SELECTED_SCREEN_INDEX
    print(f"Selected screen: {SELECTED_SCREEN_INDEX}")
    for message in sampling_loop_sync(*args, selected_screen=SELECTED_SCREEN_INDEX, **kwargs):
        if message not in accumulated_messages:
            accumulated_messages.append(message)
            yield accumulated_messages

def yield_message(state):
    if not state.get("api_key"):
        raise ValueError("API key is missing. Please set it in the environment.")
    for message in accumulate_messages(
        system_prompt_suffix=state["custom_system_prompt"],
        model=state["model"],
        provider=state["provider"],
        messages=state["messages"],
        output_callback=partial(_render_message, Sender.BOT, state=state),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
    ):
        yield message

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Supabase Computer Use")
        self.resize(800, 600)
        self.state = {}
        setup_state(self.state)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout()

        # Screen Selector
        screen_options, primary_index = get_screen_details()
        global SCREEN_NAMES, SELECTED_SCREEN_INDEX
        SCREEN_NAMES = screen_options
        SELECTED_SCREEN_INDEX = primary_index

        self.screen_selector = QComboBox()
        self.screen_selector.addItems(screen_options)
        self.screen_selector.setCurrentIndex(primary_index)
        self.screen_selector.currentIndexChanged.connect(self.update_selected_screen)

        # Slider for recent screenshots
        self.only_n_images = QSlider(Qt.Horizontal)
        self.only_n_images.setMinimum(0)
        self.only_n_images.setMaximum(10)
        self.only_n_images.setValue(2)
        self.only_n_images.setTickPosition(QSlider.TicksBelow)
        self.only_n_images.setTickInterval(1)
        self.only_n_images.valueChanged.connect(self.update_n_images)

        settings_layout.addWidget(QLabel("Select Screen:"))
        settings_layout.addWidget(self.screen_selector)
        settings_layout.addWidget(QLabel("N most recent screenshots:"))
        settings_layout.addWidget(self.only_n_images)
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # **Initialize chat_input and send_button before any method that might use them**
        # Chat Input and Send Button
        chat_input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message to send to Supabase Computer Use...")
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.process_input)

        chat_input_layout.addWidget(self.chat_input)
        chat_input_layout.addWidget(self.send_button)
        main_layout.addLayout(chat_input_layout)

        # Quick Start Prompt Group
        quick_start_group = QGroupBox("Quick Start Prompt")
        quick_start_layout = QHBoxLayout()

        # Load the merged_dict
        with open("data/examples.json", "r") as f:
            self.merged_dict = json.load(f)

        # Initialize dropdowns
        self.first_menu = QComboBox()
        self.first_menu.addItems(list(self.merged_dict.keys()))
        self.first_menu.currentIndexChanged.connect(self.update_second_menu)

        self.second_menu = QComboBox()
        self.second_menu.currentIndexChanged.connect(self.update_third_menu)

        self.third_menu = QComboBox()
        self.third_menu.currentIndexChanged.connect(self.update_textbox)

        # Image preview and hint
        self.image_preview = QLabel()
        self.image_preview.setFixedSize(200, 200)

        self.hintbox = QLabel("Task Hint: Selected options will appear here.")
        self.hintbox.setWordWrap(True)

        # Now initialize second and third menus
        self.update_second_menu()
        self.update_third_menu()

        # Layouts for dropdowns and preview
        dropdown_layout = QVBoxLayout()
        dropdown_layout.addWidget(QLabel("Task Category"))
        dropdown_layout.addWidget(self.first_menu)
        dropdown_layout.addWidget(QLabel("Software"))
        dropdown_layout.addWidget(self.second_menu)
        dropdown_layout.addWidget(QLabel("Task"))
        dropdown_layout.addWidget(self.third_menu)

        preview_layout = QVBoxLayout()
        preview_layout.addWidget(QLabel("Reference Initial State"))
        preview_layout.addWidget(self.image_preview)
        preview_layout.addWidget(self.hintbox)

        quick_start_layout.addLayout(dropdown_layout)
        quick_start_layout.addLayout(preview_layout)
        quick_start_group.setLayout(quick_start_layout)
        main_layout.addWidget(quick_start_group)

        # Chatbot History
        self.chatbot_history_area = QScrollArea()
        self.chatbot_history_area.setWidgetResizable(True)
        self.chatbot_history_widget = QWidget()
        self.chatbot_history_layout = QVBoxLayout()
        self.chatbot_history_widget.setLayout(self.chatbot_history_layout)
        self.chatbot_history_area.setWidget(self.chatbot_history_widget)
        main_layout.addWidget(self.chatbot_history_area)

        self.setLayout(main_layout)

    def update_selected_screen(self, index):
        global SELECTED_SCREEN_INDEX
        SELECTED_SCREEN_INDEX = index
        print(f"Selected screen updated to: {SELECTED_SCREEN_INDEX}")

    def update_n_images(self, value):
        self.state["only_n_most_recent_images"] = value

    def update_second_menu(self):
        selected_category = self.first_menu.currentText()
        second_options = list(self.merged_dict.get(selected_category, {}).keys())
        self.second_menu.clear()
        self.second_menu.addItems(second_options)
        self.update_third_menu()

    def update_third_menu(self):
        selected_category = self.first_menu.currentText()
        selected_software = self.second_menu.currentText()
        third_options = list(self.merged_dict.get(selected_category, {}).get(selected_software, {}).keys())
        self.third_menu.clear()
        self.third_menu.addItems(["Please select a task"] + third_options)
        self.update_textbox()

    def update_textbox(self):
        selected_category = self.first_menu.currentText()
        selected_software = self.second_menu.currentText()
        selected_task = self.third_menu.currentText()
        task_data = self.merged_dict.get(selected_category, {}).get(selected_software, {}).get(selected_task, {})
        prompt = task_data.get("prompt", "")
        preview_image_path = task_data.get("initial_state", "")
        task_hint = "Task Hint: " + task_data.get("hint", "")
        self.chat_input.setText(prompt)
        self.hintbox.setText(task_hint)

        # Load and display the image preview if available
        if preview_image_path and os.path.exists(preview_image_path):
            pixmap = QPixmap(preview_image_path)
            pixmap = pixmap.scaled(self.image_preview.size(), Qt.KeepAspectRatio)
            self.image_preview.setPixmap(pixmap)
        else:
            self.image_preview.clear()

    def process_input(self):
        user_input = self.chat_input.text()
        if not user_input.strip():
            return

        # Append user input to chat history
        user_label = QLabel(f"User: {user_input}")
        user_label.setWordWrap(True)
        user_label.setAlignment(Qt.AlignLeft)
        self.chatbot_history_layout.addWidget(user_label)
        self.chatbot_history_area.verticalScrollBar().setValue(
            self.chatbot_history_area.verticalScrollBar().maximum()
        )
        self.chat_input.clear()

        # Start a worker thread to process input without freezing the UI
        self.thread = QThread()
        self.worker = Worker(user_input, self.state)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.message_received.connect(self.display_bot_message)
        self.thread.start()

    def display_bot_message(self, message):
        if isinstance(message, bytes):
            image = QImage.fromData(message)
            pixmap = QPixmap.fromImage(image)
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            self.chatbot_history_layout.addWidget(image_label)
        else:
            bot_label = QLabel(f"Bot: {message}")
            bot_label.setWordWrap(True)
            bot_label.setAlignment(Qt.AlignRight)
            self.chatbot_history_layout.addWidget(bot_label)
        # Scroll to bottom
        self.chatbot_history_area.verticalScrollBar().setValue(
            self.chatbot_history_area.verticalScrollBar().maximum()
        )

class Worker(QThread):
    message_received = pyqtSignal(object)  # Use object to allow bytes and str
    finished = pyqtSignal()

    def __init__(self, user_input, state):
        super().__init__()
        self.user_input = user_input
        self.state = state

    def run(self):
        # Append the user input to the messages in the state
        self.state["messages"].append(
            {
                "role": Sender.USER,
                "content": [TextBlock(type="text", text=self.user_input)],
            }
        )

        # Run the sampling loop synchronously and yield messages
        for messages in yield_message(self.state):
            for message in messages:
                rendered_message = _render_message(Sender.BOT, message, self.state)
                if rendered_message:
                    # Emit the message to the main thread
                    self.message_received.emit(rendered_message)
        self.finished.emit()

def main():
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
