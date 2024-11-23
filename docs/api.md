I am trying to refactor code to be used as an API. Right now, i have coded out a repo with a PyQt5 GUI, but we are moving the frontend to a web extension in chrome. To do this, we will need to set up an API for the Chrome extension to call. Help me refactor this code to become an APi endpoint. I am using FastAPI and Uvicorn.

Here is the old PyQt5 entrypoint:

```py
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
```

Here is the loop that we are calling:

```py
"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import (
    ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from anthropic.types.beta import BetaMessage

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from scu.autopc.actor.anthropic_actor import AnthropicActor
from scu.autopc.executor.anthropic_executor import AnthropicExecutor


BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

import base64
from PIL import Image
from io import BytesIO
def decode_base64_image_and_save(base64_str):
    # 移除base64字符串的前缀（如果存在）
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    # 解码base64字符串并将其转换为PIL图像
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    # 保存图像为screenshot.png
    import datetime
    image.save(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    print("screenshot saved")
    return f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"

def decode_base64_image(base64_str):
    # 移除base64字符串的前缀（如果存在）
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    # 解码base64字符串并将其转换为PIL图像
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


def sampling_loop_sync(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlock], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[APIResponse[BetaMessage]], None],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    selected_screen: int = 0
):
    """
    Synchronous agentic sampling loop for the assistant/tool interaction of computer use.
    """
    if model == "claude-3-5-sonnet-20241022":
        # Register Actor and Executor
        actor = AnthropicActor(
            model=model,
            provider=provider,
            system_prompt_suffix=system_prompt_suffix, 
            api_key=api_key, 
            api_response_callback=api_response_callback,
            max_tokens=max_tokens,
            only_n_most_recent_images=only_n_most_recent_images,
            selected_screen=selected_screen
        )

        # Register Executor: Function of the Executor is to send messages to ChatRoom or Execute the Action
        executor = AnthropicExecutor(
            output_callback=output_callback,
            tool_output_callback=tool_output_callback,
            selected_screen=selected_screen
        )
    else:
        raise ValueError(f"Model {model} not supported")
    

    print("Start the loop")
    while True:
        # from IPython.core.debugger import Pdb; Pdb().set_trace()
        response = actor(messages=messages)

        # Example Action: BetaMessage(id='msg_01FsYVD9PkwPo6Q9vDa2SASb', content=[BetaTextBlock(text="I'll help you open a new tab. First, I'll check if a browser window is already open by taking a screenshot, and then proceed to open a new tab.", type='text'), BetaToolUseBlock(id='toolu_01C9MQvdzehkv457iee8T8M1', input={'action': 'screenshot'}, name='computer', type='tool_use')], model='claude-3-5-sonnet-20241022', role='assistant', stop_reason='tool_use', stop_sequence=None, type='message', usage=BetaUsage(cache_creation_input_tokens=None, cache_read_input_tokens=None, input_tokens=2157, output_tokens=90))
        for message, tool_result_content in executor(response, messages):
            yield message
    
        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})

def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 2, # 10
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text

```


In here we are using the anthropic_actor aswell:

```python
"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import (
    ToolResultBlockParam,
)
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from anthropic.types.beta import BetaMessage

from ...tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult


BETA_FLAG = "computer-use-2024-10-22"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    APIProvider.BEDROCK: "anthropic.claude-3-5-sonnet-20241022-v2:0",
    APIProvider.VERTEX: "claude-3-5-sonnet-v2@20241022",
}


# Check OS

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""


class AnthropicActor:
    def __init__(
        self, 
        model: str, 
        provider: APIProvider, 
        system_prompt_suffix: str, 
        api_key: str,
        api_response_callback: Callable[[APIResponse[BetaMessage]], None],
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        selected_screen: int = 0,
    ):
        self.model = model
        self.provider = provider
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images

        self.tool_collection = ToolCollection(
            ComputerTool(selected_screen=selected_screen),
            BashTool(),
            EditTool(),
        )

        self.system = (
            f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
        )

        # Instantiate the appropriate API client based on the provider
        if provider == APIProvider.ANTHROPIC:
            self.client = Anthropic(api_key=api_key)
        elif provider == APIProvider.VERTEX:
            self.client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            self.client = AnthropicBedrock()

    def __call__(
        self, 
        *,
        messages: list[BetaMessageParam]
    ):
        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, self.only_n_most_recent_images)

        # Call the API synchronously
        raw_response = self.client.beta.messages.with_raw_response.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=self.system,
            tools=self.tool_collection.to_params(),
            betas=["computer-use-2024-10-22"],
        )

        self.api_response_callback(cast(APIResponse[BetaMessage], raw_response))

        # from IPython.core.debugger import Pdb
        # Pdb().set_trace()

        response = raw_response.parse()

        return response


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content
```

We also have the executor:


```python
import asyncio
from typing import Any, Dict, cast
from collections.abc import Callable
from anthropic.types.beta import (
    BetaContentBlock,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from ...tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult


class AnthropicExecutor:
    def __init__(
        self, 
        output_callback: Callable[[BetaContentBlockParam], None], 
        tool_output_callback: Callable[[Any, str], None],
        selected_screen: int = 0
    ):
        self.tool_collection = ToolCollection(
            ComputerTool(selected_screen=selected_screen),
            BashTool(),
            EditTool(),
        )
        self.output_callback = output_callback
        self.tool_output_callback = tool_output_callback

    def __call__(self, response: BetaMessage, messages: list[BetaMessageParam]):
        new_message = {
            "role": "assistant",
            "content": cast(list[BetaContentBlockParam], response.content),
        }
        if new_message not in messages:
            messages.append(new_message)
        else:
            print("new_message already in messages, there are duplicates.")
        
        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in cast(list[BetaContentBlock], response.content):
            self.output_callback(content_block)

            # Execute the tool
            if content_block.type == "tool_use":
                # Run the asynchronous tool execution in a synchronous context
                result = asyncio.run(self.tool_collection.run(
                    name=content_block.name,
                    tool_input=cast(dict[str, Any], content_block.input),
                ))
                tool_result_content.append(
                    _make_api_tool_result(result, content_block.id)
                )
                self.tool_output_callback(result, content_block.id)

            # Craft messages based on the content_block
            # Note: to display the messages in the gradio, you should organize the messages in the following way (user message, bot message)
            display_messages = _message_display_callback(messages)
            
            # Send the messages to the gradio
            for user_msg, bot_msg in display_messages:
                yield [user_msg, bot_msg], tool_result_content

        if not tool_result_content:
            return messages
        
        return tool_result_content

def _message_display_callback(messages):
    display_messages = []
    for msg in messages:
        try:
            if isinstance(msg["content"][0], TextBlock):
                display_messages.append((msg["content"][0].text, None))  # User message
            elif isinstance(msg["content"][0], BetaTextBlock):
                display_messages.append((None, msg["content"][0].text))  # Bot message
            elif isinstance(msg["content"][0], BetaToolUseBlock):
                display_messages.append((None, f"Tool Use: {msg['content'][0].name}\nInput: {msg['content'][0].input}"))  # Bot message
            elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "image":
                display_messages.append((None, f'<img src="data:image/png;base64,{msg["content"][0]["content"][-1]["source"]["data"]}">'))  # Bot message
            elif isinstance(msg["content"][0], Dict) and msg["content"][0]["content"][-1]["type"] == "text":
                # image_path = decode_base64_image_and_save(msg["content"][0]["content"][-1]["source"]["data"])
                # res.append((None, gr.Image(image_path)))  # Bot message
                display_messages.append((None, msg["content"][0]["content"][-1]["text"]))  # Bot message
            else:
                print(msg["content"][0])
        except Exception as e:
            print("error", e)
            pass
    return display_messages

def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
```


In terms of tools, here are all of them:

```python
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any

from anthropic.types.beta import BetaToolUnionParam


class BaseAnthropicTool(metaclass=ABCMeta):
    """Abstract base class for Anthropic-defined tools."""

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Executes the tool with the given arguments."""
        ...

    @abstractmethod
    def to_params(
        self,
    ) -> BetaToolUnionParam:
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """Represents the result of a tool execution."""

    output: str | None = None
    error: str | None = None
    base64_image: str | None = None
    system: str | None = None

    def __bool__(self):
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""


class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


import asyncio
import os
from typing import ClassVar, Literal

from anthropic.types.beta import BetaToolBash20241022Param

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult


class _BashSession:
    """A session of a bash shell."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            shell=False,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """Terminate the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """Execute a command in the bash shell."""
        if not self._started:
            raise ToolError("Session has not started.")
        if self._process.returncode is not None:
            return ToolResult(
                system="tool must be restarted",
                error=f"bash has exited with returncode {self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        output = ""
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    data = await self._process.stdout.readline()
                    if not data:
                        break
                    line = data.decode()
                    output += line
                    if self._sentinel in line:
                        output = output.replace(self._sentinel, "")
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"timed out: bash has not returned in {self._timeout} seconds and must be restarted",
            ) from None

        error = await self._process.stderr.read()
        error = error.decode()

        return CLIResult(output=output.strip(), error=error.strip())


class BashTool(BaseAnthropicTool):
    """
    A tool that allows the agent to run bash commands.
    The tool parameters are defined by Anthropic and are not editable.
    """

    _session: _BashSession | None
    name: ClassVar[Literal["bash"]] = "bash"
    api_type: ClassVar[Literal["bash_20241022"]] = "bash_20241022"

    def __init__(self):
        self._session = None
        super().__init__()

    async def __call__(
        self, command: str | None = None, restart: bool = False, **kwargs
    ):
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return ToolResult(system="tool has been restarted.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("no command provided.")

    def to_params(self) -> BetaToolBash20241022Param:
        return {
            "type": self.api_type,
            "name": self.name,
        }
    
"""Collection classes for managing multiple tools."""

from typing import Any

from anthropic.types.beta import BetaToolUnionParam

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)


class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(self, *tools: BaseAnthropicTool):
        self.tools = tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in tools}

    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)

        import subprocess
import platform
import pyautogui
import asyncio
import base64
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4
from screeninfo import get_monitors

from PIL import ImageGrab, Image
from functools import partial

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import run

OUTPUT_DIR = "./tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
]


class Resolution(TypedDict):
    width: int
    height: int


MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def get_screen_details():
    screens = get_monitors()
    screen_details = []

    # Sort screens by x position to arrange from left to right
    sorted_screens = sorted(screens, key=lambda s: s.x)

    # Loop through sorted screens and assign positions
    primary_index = 0
    for i, screen in enumerate(sorted_screens):
        if i == 0:
            layout = "Left"
        elif i == len(sorted_screens) - 1:
            layout = "Right"
        else:
            layout = "Center"
        
        if screen.is_primary:
            position = "Primary" 
            primary_index = i
        else:
            position = "Secondary"
        screen_info = f"Screen {i + 1}: {screen.width}x{screen.height}, {layout}, {position}"
        screen_details.append(screen_info)

    return screen_details, primary_index


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    Adapted for Windows using 'pyautogui'.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, selected_screen: int = 0):
        super().__init__()

        # Get screen width and height using Windows command
        self.display_num = None
        self.offset_x = 0
        self.offset_y = 0
        self.selected_screen = selected_screen   
        self.width, self.height = self.get_screen_size()     

        # Path to cliclick
        self.cliclick = "cliclick"
        self.key_conversion = {"Page_Down": "pagedown", "Page_Up": "pageup", "Super_L": "win"}

    async def __call__(
        self,
        *,
        action: Action,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        **kwargs,
    ):
        print(f"action: {action}, text: {text}, coordinate: {coordinate}")
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )
            x += self.offset_x
            y += self.offset_y

            if action == "mouse_move":
                pyautogui.moveTo(x, y)
                return ToolResult(output=f"Moved mouse to ({x}, {y})")
            elif action == "left_click_drag":
                current_x, current_y = pyautogui.position()
                pyautogui.dragTo(x, y, duration=0.5)  # Adjust duration as needed
                return ToolResult(output=f"Dragged mouse from ({current_x}, {current_y}) to ({x}, {y})")

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(output=f"{text} must be a string")

            if action == "key":
                # Handle key combinations
                keys = text.split('+')
                for key in keys:
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    pyautogui.keyDown(key)  # Press down each key
                for key in reversed(keys):
                    key = self.key_conversion.get(key.strip(), key.strip())
                    key = key.lower()
                    pyautogui.keyUp(key)    # Release each key in reverse order
                return ToolResult(output=f"Pressed keys: {text}")
            
            elif action == "type":
                pyautogui.typewrite(text, interval=TYPING_DELAY_MS / 1000)  # Convert ms to seconds
                screenshot_base64 = (await self.screenshot()).base64_image
                return ToolResult(output=text, base64_image=screenshot_base64)

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return await self.screenshot()
            elif action == "cursor_position":
                x, y = pyautogui.position()
                x, y = self.scale_coordinates(ScalingSource.COMPUTER, x, y)
                return ToolResult(output=f"X={x},Y={y}")
            else:
                if action == "left_click":
                    pyautogui.click()
                elif action == "right_click":
                    pyautogui.rightClick()
                elif action == "middle_click":
                    pyautogui.middleClick()
                elif action == "double_click":
                    pyautogui.doubleClick()
                return ToolResult(output=f"Performed {action}")
        raise ToolError(f"Invalid action: {action}")

    async def screenshot(self):
        """Take a screenshot of the current screen and return a ToolResult with the base64 encoded image."""
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"screenshot_{uuid4().hex}.png"

        ImageGrab.grab = partial(ImageGrab.grab, all_screens=True)

        # Detect platform
        system = platform.system()

        if system == "Windows":
            # Windows: Use screeninfo to get monitor details
            screens = get_monitors()

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s.x)

            if self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[self.selected_screen]
            bbox = (screen.x, screen.y, screen.x + screen.width, screen.y + screen.height)

        elif system == "Darwin":  # macOS
            # macOS: Use Quartz to get monitor details
            max_displays = 32  # Maximum number of displays to handle
            active_displays = Quartz.CGGetActiveDisplayList(max_displays, None, None)[1]

            # Get the display bounds (resolution) for each active display
            screens = []
            for display_id in active_displays:
                bounds = Quartz.CGDisplayBounds(display_id)
                screens.append({
                    'id': display_id,
                    'x': int(bounds.origin.x),
                    'y': int(bounds.origin.y),
                    'width': int(bounds.size.width),
                    'height': int(bounds.size.height),
                    'is_primary': Quartz.CGDisplayIsMain(display_id)  # Check if this is the primary display
                })

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s['x'])

            if self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")

            screen = sorted_screens[self.selected_screen]
            bbox = (screen['x'], screen['y'], screen['x'] + screen['width'], screen['y'] + screen['height'])

        else:  # Linux or other OS
            cmd = "xrandr | grep ' primary' | awk '{print $4}'"
            try:
                output = subprocess.check_output(cmd, shell=True).decode()
                resolution = output.strip().split()[0]
                width, height = map(int, resolution.split('x'))
                bbox = (0, 0, width, height)  # Assuming single primary screen for simplicity
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to get screen resolution on Linux.")

        # Take screenshot using the bounding box
        screenshot = ImageGrab.grab(bbox=bbox)

        # Set offsets (for potential future use)
        self.offset_x = screen['x'] if system == "Darwin" else screen.x
        self.offset_y = screen['y'] if system == "Darwin" else screen.y

        if not hasattr(self, 'target_dimension'):
            screenshot = self.padding_image(screenshot)
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]

        # Resize if target_dimensions are specified
        print(f"offset is {self.offset_x}, {self.offset_y}")
        print(f"target_dimension is {self.target_dimension}")
        screenshot = screenshot.resize((self.target_dimension["width"], self.target_dimension["height"]))


        # Save the screenshot
        screenshot.save(str(path))

        if path.exists():
            # Return a ToolResult instance instead of a dictionary
            return ToolResult(base64_image=base64.b64encode(path.read_bytes()).decode())
        
        raise ToolError(f"Failed to take screenshot: {path} does not exist.")

    def padding_image(self, screenshot):
        """Pad the screenshot to 16:10 aspect ratio, when the aspect ratio is not 16:10."""
        _, height = screenshot.size
        new_width = height * 16 // 10

        padding_image = Image.new("RGB", (new_width, height), (255, 255, 255))
        # padding to top left
        padding_image.paste(screenshot, (0, 0))
        return padding_image

    async def shell(self, command: str, take_screenshot=True) -> ToolResult:
        """Run a shell command and return the output, error, and optionally a screenshot."""
        _, stdout, stderr = await run(command)
        base64_image = None

        if take_screenshot:
            # delay to let things settle before taking a screenshot
            await asyncio.sleep(self._screenshot_delay)
            base64_image = (await self.screenshot()).base64_image

        return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None

        for target_name, dimension in MAX_SCALING_TARGETS.items():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                    self.target_dimension = target_dimension
                    # print(f"target_dimension: {target_dimension}")
                break

        if target_dimension is None:
            # TODO: currently we force the target to be WXGA (16:10), when it cannot find a match
            target_dimension = MAX_SCALING_TARGETS["WXGA"]
            self.target_dimension = MAX_SCALING_TARGETS["WXGA"]

        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)

    def get_screen_size(self):
        if platform.system() == "Windows":
            # Use screeninfo to get primary monitor on Windows
            screens = get_monitors()

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s.x)
            
            if self.selected_screen is None:
                primary_monitor = next((m for m in get_monitors() if m.is_primary), None)
                return primary_monitor.width, primary_monitor.height
            elif self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")
            else:
                screen = sorted_screens[self.selected_screen]
                return screen.width, screen.height

        elif platform.system() == "Darwin":
            # macOS part using Quartz to get screen information
            max_displays = 32  # Maximum number of displays to handle
            active_displays = Quartz.CGGetActiveDisplayList(max_displays, None, None)[1]

            # Get the display bounds (resolution) for each active display
            screens = []
            for display_id in active_displays:
                bounds = Quartz.CGDisplayBounds(display_id)
                screens.append({
                    'id': display_id,
                    'x': int(bounds.origin.x),
                    'y': int(bounds.origin.y),
                    'width': int(bounds.size.width),
                    'height': int(bounds.size.height),
                    'is_primary': Quartz.CGDisplayIsMain(display_id)  # Check if this is the primary display
                })

            # Sort screens by x position to arrange from left to right
            sorted_screens = sorted(screens, key=lambda s: s['x'])

            if self.selected_screen is None:
                # Find the primary monitor
                primary_monitor = next((screen for screen in screens if screen['is_primary']), None)
                if primary_monitor:
                    return primary_monitor['width'], primary_monitor['height']
                else:
                    raise RuntimeError("No primary monitor found.")
            elif self.selected_screen < 0 or self.selected_screen >= len(screens):
                raise IndexError("Invalid screen index.")
            else:
                # Return the resolution of the selected screen
                screen = sorted_screens[self.selected_screen]
                return screen['width'], screen['height']

        else:  # Linux or other OS
            cmd = "xrandr | grep ' primary' | awk '{print $4}'"
            try:
                output = subprocess.check_output(cmd, shell=True).decode()
                resolution = output.strip().split()[0]
                width, height = map(int, resolution.split('x'))
                return width, height
            except subprocess.CalledProcessError:
                raise RuntimeError("Failed to get screen resolution on Linux.")
    
    def get_mouse_position(self):
        # TODO: enhance this func
        from AppKit import NSEvent
        from Quartz import CGEventSourceCreate, kCGEventSourceStateCombinedSessionState

        loc = NSEvent.mouseLocation()
        # Adjust for different coordinate system
        return int(loc.x), int(self.height - loc.y)

    def map_keys(self, text: str):
        """Map text to cliclick key codes if necessary."""
        # For simplicity, return text as is
        # Implement mapping if special keys are needed
        return text
from collections import defaultdict
from pathlib import Path
from typing import Literal, get_args

from anthropic.types.beta import BetaToolTextEditor20241022Param

from .base import BaseAnthropicTool, CLIResult, ToolError, ToolResult
from .run import maybe_truncate, run

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4


class EditTool(BaseAnthropicTool):
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    api_type: Literal["text_editor_20241022"] = "text_editor_20241022"
    name: Literal["str_replace_editor"] = "str_replace_editor"

    _file_history: dict[Path, list[str]]

    def __init__(self):
        self._file_history = defaultdict(list)
        super().__init__()

    def to_params(self) -> BetaToolTextEditor20241022Param:
        return {
            "name": self.name,
            "type": self.api_type,
        }

    async def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
    ):
        _path = Path(path)
        self.validate_path(command, _path)
        if command == "view":
            return await self.view(_path, view_range)
        elif command == "create":
            if not file_text:
                raise ToolError("Parameter `file_text` is required for command: create")
            self.write_file(_path, file_text)
            self._file_history[_path].append(file_text)
            return ToolResult(output=f"File created successfully at: {_path}")
        elif command == "str_replace":
            if not old_str:
                raise ToolError(
                    "Parameter `old_str` is required for command: str_replace"
                )
            return self.str_replace(_path, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                raise ToolError(
                    "Parameter `insert_line` is required for command: insert"
                )
            if not new_str:
                raise ToolError("Parameter `new_str` is required for command: insert")
            return self.insert(_path, insert_line, new_str)
        elif command == "undo_edit":
            return self.undo_edit(_path)
        raise ToolError(
            f'Unrecognized command {command}. The allowed commands for the {self.name} tool are: {", ".join(get_args(Command))}'
        )

    def validate_path(self, command: str, path: Path):
        """
        Check that the path/command combination is valid.
        """
        # Check if its an absolute path
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ToolError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )
        # Check if path exists
        if not path.exists() and command != "create":
            raise ToolError(
                f"The path {path} does not exist. Please provide a valid path."
            )
        if path.exists() and command == "create":
            raise ToolError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        # Check if the path points to a directory
        if path.is_dir():
            if command != "view":
                raise ToolError(
                    f"The path {path} is a directory and only the `view` command can be used on directories"
                )

    async def view(self, path: Path, view_range: list[int] | None = None):
        """Implement the view command"""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            _, stdout, stderr = await run(
                rf"find {path} -maxdepth 2 -not -path '*/\.*'"
            )
            if not stderr:
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            return CLIResult(output=stdout, error=stderr)

        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError(
                    "Invalid `view_range`. It should be a list of two integers."
                )
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. It's first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. It's second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. It's second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return CLIResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    def str_replace(self, path: Path, old_str: str, new_str: str | None):
        """Implement the str_replace command, which replaces old_str with new_str in the file content"""
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        self.write_file(path, new_file_content)

        # Save the content to history
        self._file_history[path].append(file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return CLIResult(output=success_msg)

    def insert(self, path: Path, insert_line: int, new_str: str):
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return CLIResult(output=success_msg)

    def undo_edit(self, path: Path):
        """Implement the undo_edit command."""
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        return CLIResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path):
        """Read the content of a file from a given path; raise a ToolError if an error occurs."""
        try:
            return path.read_text()
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        try:
            path.write_text(file)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to write to {path}") from None

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )

    
    
    
"""Utility to run shell commands asynchronously with a timeout."""

import asyncio

TRUNCATED_MESSAGE: str = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
MAX_RESPONSE_LEN: int = 16000


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


async def run(
    cmd: str,
    timeout: float | None = 120.0,  # seconds
    truncate_after: int | None = MAX_RESPONSE_LEN,
):
    """Run a shell command asynchronously with a timeout."""
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        return (
            process.returncode or 0,
            maybe_truncate(stdout.decode(), truncate_after=truncate_after),
            maybe_truncate(stderr.decode(), truncate_after=truncate_after),
        )
    except asyncio.TimeoutError as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise TimeoutError(
            f"Command '{cmd}' timed out after {timeout} seconds"
        ) from exc



```


Think deeply on how we can convert this to an APi, and how this will be called form the chrome extension. Then code out the API endpoint for me. Here is a starter:

```python router.py
from uuid import uuid4
from fastapi import HTTPException, Depends, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from api.core.auth import verify_token
from api.utils.constants import SHARED
from typing import Dict, Any, List, Optional

import json
from fastapi import HTTPException, APIRouter
from typing import Dict, Any, List
from uuid import uuid4

from .computer_use import (
    setup_state,
    AnthropicActor,
    MessageRequest,
    MessageResponse,
    ContentBlock,
)

from uuid import uuid4
from fastapi import HTTPException, APIRouter
from typing import Dict, Any, List
import json

from .computer_use import (
    setup_state,
    AnthropicActor,
    MessageRequest,
    MessageResponse,
    ContentBlock,
)

router = APIRouter()

# In-memory store for conversation states
conversation_states: Dict[str, Dict[str, Any]] = {}

@router.post("/agent")
async def agent_endpoint(request: MessageRequest):
    conversation_id = request.conversation_id or str(uuid4())

    # Retrieve or create the conversation state
    if conversation_id in conversation_states:
        state = conversation_states[conversation_id]
    else:
        state = {}
        setup_state(state)
        conversation_states[conversation_id] = state

    # Add user message to messages
    state["messages"].append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    try:
        # Create the actor
        actor = AnthropicActor(
            model=state["model"],
            system_prompt=state["system_prompt"],
            api_key=state["api_key"],
            max_tokens=4096,
        )

        assistant_response = actor(messages=state["messages"])

        # Add assistant's message to messages
        state["messages"].append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )

        content_blocks = []

        # Try to parse the assistant's response as JSON
        try:
            command = json.loads(assistant_response)
            content_blocks.append(ContentBlock(type="command", content=command))
        except json.JSONDecodeError:
            # Not a command, treat as normal text
            content_blocks.append(ContentBlock(type="text", content={"text": assistant_response}))

        return MessageResponse(conversation_id=conversation_id, content=content_blocks)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent-rag")
async def search_companies_endpoint(
        background_tasks: BackgroundTasks,
        query: str
):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty.")

    try:
        agent = SHARED["extraction_agent"]
        supabase = SHARED["supabase_client"]
        result = agent.invoke(query)

        # Add new companies to the Supabase
        # background_tasks.add_task(add_apollo_companies_supabase, supabase, companies)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


```

Here is the `computer_use.py`:

```python
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel
from anthropic import Anthropic

from anthropic.types.beta import (

    BetaMessage,
    BetaMessageParam,
)

from api.scu.tools.collection import ToolCollection
from api.scu.tools.computer import ComputerTool

BETA_FLAG = "computer-use-2024-10-22"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

# Helper functions
def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "api_key" not in state:
        state["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
        if not state["api_key"]:
            raise ValueError("API key not found. Please set it in the environment.")
    if "model" not in state:
        state["model"] = "claude-3-5-sonnet-20241022"  # Updated model
    if "system_prompt_suffix" not in state:
        device_os_name = (
            "Windows"
            if os.name == "nt"
            else "Mac"
            if os.uname().sysname == "Darwin"
            else "Linux"
        )
        state["system_prompt_suffix"] = f"\n\nNOTE: you are operating a {device_os_name} machine. The current date is {datetime.today().strftime('%A, %B %d, %Y')}."
    if "responses" not in state:
        state["responses"] = {}


# Pydantic models for request and response
class MessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class ContentBlock(BaseModel):
    type: str
    content: Dict[str, Any]

class MessageResponse(BaseModel):
    conversation_id: str
    content: List[ContentBlock]

from typing import Callable

class AnthropicActor:
    def __init__(
        self,
        model: str,
        system_prompt_suffix: str,
        api_key: str,
        api_response_callback: Callable[[APIResponse[BetaMessage]], None],
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        selected_screen: int = 0,
    ):
        self.model = model
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images

        # Since we're removing local tools, we'll not define them here
        self.tool_collection = ToolCollection()  # Empty ToolCollection

        self.system = (
            f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
        )

        # Instantiate the Anthropic API client
        self.client = Anthropic(api_key=api_key)

    def __call__(
        self,
        *,
        messages: list[BetaMessageParam]
    ):
        if self.only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(messages, self.only_n_most_recent_images)

        # Call the API synchronously
        raw_response = self.client.beta.messages.with_raw_response.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=self.system,
            tools=self.tool_collection.to_params(),
            betas=[BETA_FLAG],
        )

        self.api_response_callback(cast(APIResponse[BetaMessage], raw_response))

        response = raw_response.parse()

        return response


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int = 10,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[ToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content
```

refactor the provided code. We are trying to remove PyQt5 GUI and put the logic into a FastAPI endpoint that can be called from a Chrome extension. The goal is to create an API that accepts user messages, processes them through the existing logic (including interactions with the Anthropic API and tools), and returns the assistant's response.

 The original code includes tools that interact with the local system (e.g., taking screenshots and clicking). This will be done by the chrome extension now, so we do not need this logic on the server.
