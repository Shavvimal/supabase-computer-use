The recently released model, Claude 3.5 Computer Use, stands out as the first frontier AI model to offer computer use in public beta as a graphical user interface (GUI) agent. I will provide you code from a case study to explore Claude 3.5 Computer Use. They curate and organize a collection of carefully designed tasks spanning a variety of domains and software. They provide an out-of-the-box agent framework for deploying API-based GUI automation models with easy implementation.

How computer use works

1. Provide Claude with computer use tools and a user prompt

Add Anthropic-defined computer use tools to your API request.
Include a user prompt that might require these tools, e.g., “Save a picture of a cat to my desktop.” 2. Claude decides to use a tool

Claude loads the stored computer use tool definitions and assesses if any tools can help with the user’s query.
If yes, Claude constructs a properly formatted tool use request.
The API response has a stop_reason of tool_use, signaling Claude’s intent. 3. Extract tool input, evaluate the tool on a computer, and return results

On your end, extract the tool name and input from Claude’s request.
Use the tool on a container or Virtual Machine.
Continue the conversation with a new user message containing a tool_result content block. 4. Claude continues calling computer use tools until it's completed the task

Claude analyzes the tool results to determine if more tool use is needed or the task has been completed.
If Claude decides it needs another tool, it responds with another tool_use stop_reason and you should return to step 3.
Otherwise, it crafts a text response to the user.
We refer to the repetition of steps 3 and 4 without user input as the “agent loop” - i.e., Claude responding with a tool use request and your application responding to Claude with the results of evaluating that request.

I am trying to refactor code to be used as an API. Right now, i have coded out a repo with a Gragio GUI, but we are moving the frontend to a web extension in chrome. To do this, we will need to set up an API for the Chrome extension to call. Help me refactor this code to become an APi endpoint. I am using FastAPI and Uvicorn. Here is the old Gradio entrypoint:

```py
"""
Entrypoint for Gradio, see https://gradio.app/
"""

import platform
import asyncio
import base64
import os
import json
from datetime import datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import cast, Dict

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock

from screeninfo import get_monitors

# TODO: I don't know why If don't get monitors here, the screen resolution will be wrong for secondary screen. Seems there are some conflict with computer_use_demo.tools
screens = get_monitors()
print(screens)
from computer_use_demo.loop import (
    PROVIDER_TO_DEFAULT_MODEL_NAME,
    APIProvider,
    # sampling_loop,
    sampling_loop_sync,
)

from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.computer import get_screen_details

CONFIG_DIR = Path("~/.anthropic").expanduser()
API_KEY_FILE = CONFIG_DIR / "api_key"

WARNING_TEXT = "⚠️ Security Alert: Never provide access to sensitive accounts or data, as malicious web content can hijack Claude's behavior"

SELECTED_SCREEN_INDEX = None
SCREEN_NAMES = None

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "api_key" not in state:
        # Try to load API key from file first, then environment
        state["api_key"] = load_from_storage("api_key") or os.getenv("ANTHROPIC_API_KEY", "")
        if not state["api_key"]:
            print("API key not found. Please set it in the environment or storage.")
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
        state["only_n_most_recent_images"] = 2 # 10
    if "custom_system_prompt" not in state:
        state["custom_system_prompt"] = load_from_storage("system_prompt") or ""
        # remove if want to use default system prompt
        device_os_name = "Windows" if platform.platform() == "Windows" else "Mac" if platform.platform() == "Darwin" else "Linux"
        state["custom_system_prompt"] += f"\n\nNOTE: you are operating a {device_os_name} machine"
    if "hide_images" not in state:
        state["hide_images"] = False


def _reset_model(state):
    state["model"] = PROVIDER_TO_DEFAULT_MODEL_NAME[cast(APIProvider, state["provider"])]


async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"


def validate_auth(provider: APIProvider, api_key: str | None):
    if provider == APIProvider.ANTHROPIC:
        if not api_key:
            return "Enter your Anthropic API key to continue."
    if provider == APIProvider.BEDROCK:
        import boto3

        if not boto3.Session().get_credentials():
            return "You must have AWS credentials set up to use the Bedrock API."
    if provider == APIProvider.VERTEX:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        if not os.environ.get("CLOUD_ML_REGION"):
            return "Set the CLOUD_ML_REGION environment variable to use the Vertex API."
        try:
            google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        except DefaultCredentialsError:
            return "Your google cloud credentials are not set up correctly."


def load_from_storage(filename: str) -> str | None:
    """Load data from a file in the storage directory."""
    try:
        file_path = CONFIG_DIR / filename
        if file_path.exists():
            data = file_path.read_text().strip()
            if data:
                return data
    except Exception as e:
        print(f"Debug: Error loading {filename}: {e}")
    return None


def save_to_storage(filename: str, data: str) -> None:
    """Save data to a file in the storage directory."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = CONFIG_DIR / filename
        file_path.write_text(data)
        # Ensure only user can read/write the file
        file_path.chmod(0o600)
    except Exception as e:
        print(f"Debug: Error saving {filename}: {e}")


def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response


def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output


def _render_message(sender: Sender, message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, state):
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
# open new tab, open google sheets inside, then create a new blank spreadsheet

def process_input(user_input, state):
    # Ensure the state is properly initialized
    setup_state(state)

    # Append the user input to the messages in the state
    state["messages"].append(
        {
            "role": Sender.USER,
            "content": [TextBlock(type="text", text=user_input)],
        }
    )

    # Run the sampling loop synchronously and yield messages
    for message in yield_message(state):
        yield message


def accumulate_messages(*args, **kwargs):
    """
    Wrapper function to accumulate messages from sampling_loop_sync.
    """
    accumulated_messages = []
    global SELECTED_SCREEN_INDEX
    print(f"Selected screen: {SELECTED_SCREEN_INDEX}")
    for message in sampling_loop_sync(*args, selected_screen=SELECTED_SCREEN_INDEX, **kwargs):
        # Check if the message is already in the accumulated messages
        if message not in accumulated_messages:
            accumulated_messages.append(message)
            # Yield the accumulated messages as a list
            yield accumulated_messages


def yield_message(state):
    # Ensure the API key is present
    if not state.get("api_key"):
        raise ValueError("API key is missing. Please set it in the environment or storage.")

    # Call the sampling loop and yield messages
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


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state = gr.State({})  # Use Gradio's state management

    # Retrieve screen details
    gr.Markdown("# Computer Use OOTB")

    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(WARNING_TEXT)

    with gr.Accordion("Settings", open=False):
        with gr.Row():
            with gr.Column():
                model = gr.Textbox(label="Model", value="claude-3-5-sonnet-20241022")
            with gr.Column():
                provider = gr.Dropdown(
                    label="API Provider",
                    choices=[option.value for option in APIProvider],
                    value="anthropic",
                    interactive=True,
                )
            with gr.Column():
                api_key = gr.Textbox(
                    label="Anthropic API Key",
                    type="password",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                custom_prompt = gr.Textbox(
                    label="System Prompt Suffix",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                screen_options, primary_index = get_screen_details()
                SCREEN_NAMES = screen_options
                SELECTED_SCREEN_INDEX = primary_index
                screen_selector = gr.Dropdown(
                    label="Select Screen",
                    choices=screen_options,
                    value=screen_options[primary_index] if screen_options else None,
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    value=2,
                    interactive=True,
                )
        # hide_images = gr.Checkbox(label="Hide screenshots", value=False)

    # Define the merged dictionary with task mappings
    merged_dict = json.load(open("examples/ootb_examples.json", "r"))

    # Callback to update the second dropdown based on the first selection
    def update_second_menu(selected_category):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).keys()))

    # Callback to update the third dropdown based on the second selection
    def update_third_menu(selected_category, selected_option):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).get(selected_option, {}).keys()))

    # Callback to update the textbox based on the third selection
    def update_textbox(selected_category, selected_option, selected_task):
        task_data = merged_dict.get(selected_category, {}).get(selected_option, {}).get(selected_task, {})
        prompt = task_data.get("prompt", "")
        preview_image = task_data.get("initial_state", "")
        task_hint = "Task Hint: " + task_data.get("hint", "")
        return prompt, preview_image, task_hint

    # Function to update the global variable when the dropdown changes
    def update_selected_screen(selected_screen_name):
        global SCREEN_NAMES
        global SELECTED_SCREEN_INDEX
        SELECTED_SCREEN_INDEX = SCREEN_NAMES.index(selected_screen_name)
        print(f"Selected screen updated to: {SELECTED_SCREEN_INDEX}")

    with gr.Accordion("Quick Start Prompt", open=False):  # open=False 表示默认收
        # Initialize Gradio interface with the dropdowns
        with gr.Row():
            # Set initial values
            initial_category = "Game Play"
            initial_second_options = list(merged_dict[initial_category].keys())
            initial_third_options = list(merged_dict[initial_category][initial_second_options[0]].keys())
            initial_text_value = merged_dict[initial_category][initial_second_options[0]][initial_third_options[0]]

            with gr.Column(scale=2):
            # First dropdown for Task Category
                first_menu = gr.Dropdown(
                    choices=list(merged_dict.keys()), label="Task Category", interactive=True, value=initial_category
                )

                # Second dropdown for Software
                second_menu = gr.Dropdown(
                    choices=initial_second_options, label="Software", interactive=True, value=initial_second_options[0]
                )

                # Third dropdown for Task
                third_menu = gr.Dropdown(
                    # choices=initial_third_options, label="Task", interactive=True, value=initial_third_options[0]
                    choices=["Please select a task"]+initial_third_options, label="Task", interactive=True, value="Please select a task"
                )

            with gr.Column(scale=1):
                image_preview = gr.Image(label="Reference Initial State", height=260 - (318.75-280))
                hintbox = gr.Markdown("Task Hint: Selected options will appear here.")


        # Textbox for displaying the mapped value
        # textbox = gr.Textbox(value=initial_text_value, label="Action")

    api_key.change(fn=lambda key: save_to_storage(API_KEY_FILE, key), inputs=api_key)

    with gr.Row():
        # submit_button = gr.Button("Submit")  # Add submit button
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Computer Use OOTB...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")

    chatbot = gr.Chatbot(label="Chatbot History", autoscroll=True, height=580)

    screen_selector.change(fn=update_selected_screen, inputs=screen_selector, outputs=None)

    # Link callbacks to update dropdowns based on selections
    first_menu.change(fn=update_second_menu, inputs=first_menu, outputs=second_menu)
    second_menu.change(fn=update_third_menu, inputs=[first_menu, second_menu], outputs=third_menu)
    third_menu.change(fn=update_textbox, inputs=[first_menu, second_menu, third_menu], outputs=[chat_input, image_preview, hintbox])

    # chat_input.submit(process_input, [chat_input, state], chatbot)
    submit_button.click(process_input, [chat_input, state], chatbot)

demo.launch(share=True)
```

Here is the loop that we are calling:

```py
"""
Agentic sampling loop that calls the Anthropic API and local implenmentation of anthropic-defined computer use tools.
"""
import asyncio
import platform
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
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock

from .tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from PIL import Image
from io import BytesIO
import gradio as gr
from typing import Dict

from computer_use_demo.autopc.actor.anthropic_actor import AnthropicActor
from computer_use_demo.autopc.executor.anthropic_executor import AnthropicExecutor


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
import asyncio
import platform
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
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock

from ...tools import BashTool, ComputerTool, EditTool, ToolCollection, ToolResult

from PIL import Image
from io import BytesIO
import gradio as gr
from typing import Dict


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

Think deeply on how we can convert this to an APi, and how this will be called form the chrome extension.

I have so far made a class that is long lived called the anthrpic actor. I have provided the executor to you because even though we do not want to execute form the API anymore, how we format messages is important. I want to keep the same message format but only execute them on the frontend. How do i do this back and forth in the same way? Adjust the conversation flow to allow the client to execute tools and send back results. The gradio app is also sending screenshots and using that to instruct where to click. How can i do that in the API?

I have instantiated tools below, but i have removed the functionality. The idea is that I want to pass `tools=self.tool_collection.to_params()` to the API call, so do not worry that the tools are there. I have removed all dependencies for interacting on the server side such as `pyautogui`

in my root file, the actor is instantiated like so:

```py main.py
from api.scu.computer_use import AnthropicActor

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = getLogger("API")
    ...

    try:
        SHARED["extraction_agent"] = AgenticRAG()
    except Exception as e:
        logger.error(f"Failed to initialize extraction_agent: {e}")

...

app = FastAPI(
  lifespan=lifespan,
  ...
)
```

```python computer_use.py
from datetime import datetime
from typing import List, Optional, Dict, Any
import os
from pydantic import BaseModel
from anthropic import Anthropic
from anthropic.types.beta import (
    BetaMessageParam,
)
from enum import StrEnum
from api.scu.tools import ToolCollection, ComputerTool

class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Browser with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "responses" not in state:
        state["responses"] = {}

class ContentBlock(BaseModel):
    type: str  # e.g., "text", "image", "tool_result", "tool_use"
    content: Dict[str, Any]

class MessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    content_blocks: List[ContentBlock]

class MessageResponse(BaseModel):
    conversation_id: str
    content: List[ContentBlock]

class AnthropicActor:
    def __init__(
        self,
        max_tokens: int = 4096,
    ):
        self.model = "claude-3-5-sonnet-20241022"
        self.system_prompt_suffix = f"\n\nNOTE: you are operating a browser. The current date is {datetime.today().strftime('%A, %B %d, %Y')}."
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.max_tokens = max_tokens

        self.system = (
            f"{SYSTEM_PROMPT}{' ' + self.system_prompt_suffix if self.system_prompt_suffix else ''}"
        )

        # Instantiate the Anthropic API client
        self.client = Anthropic(api_key=self.api_key)
        # Pass tools to the assistant but we won't execute them on the server
        self.tool_collection = ToolCollection(
            ComputerTool(),
        )

    def __call__(
        self,
        *,
        messages: List[BetaMessageParam]
    ):
        # Call the API synchronously
        raw_response = self.client.beta.messages.with_raw_response.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=self.system,
            tools=self.tool_collection.to_params(),
            betas=["computer-use-2024-10-22"],
        )

        response = raw_response.parse()

        return response
```

Here is the `router.py`:

```python
from fastapi import  BackgroundTasks
from fastapi.responses import JSONResponse
from api.utils.constants import SHARED
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from typing import Dict
from typing import Any
from .computer_use import (
    setup_state,
    MessageRequest,
    MessageResponse,
    ContentBlock,
    Sender
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

    if not request.content_blocks:
        raise HTTPException(status_code=400, detail="No content_blocks provided.")

    # Convert content_blocks to the format expected by the assistant
    user_content = []
    for block in request.content_blocks:
        if block.type == "text":
            user_content.append({
                "type": "text",
                "text": block.content.get("text", "")
            })
        elif block.type == "image":
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": block.content.get("data", "")
                }
            })
        elif block.type == "tool_result":
            user_content.append({
                "type": "tool_result",
                "tool_use_id": block.content.get("tool_use_id"),
                "is_error": block.content.get("is_error", False),
                "content": block.content.get("content", [])
            })
        else:
            # Handle other types as needed
            user_content.append(block.dict())

    # Append the user's content to the conversation
    state["messages"].append({
        "role": Sender.USER,
        "content": user_content,
    })

    try:
        # Access the actor
        anthropic_actor = SHARED["anthropic_actor"]
        assistant_response = anthropic_actor(messages=state["messages"])

        # Add assistant's message to messages
        state["messages"].append({
            "role": Sender.BOT,
            "content": assistant_response.content,
        })

        # Process assistant's response content
        content_blocks = []

        # Check the assistant's stop_reason
        if assistant_response.stop_reason == 'tool_use':
            # Assistant is requesting tool usage
            # Return the assistant's response to the client
            for content_block in assistant_response.content:
                content_type = content_block.type
                if content_type == "text":
                    content_blocks.append(ContentBlock(type="text", content={"text": content_block.text}))
                elif content_type == "tool_use":
                    content_blocks.append(ContentBlock(type="tool_use", content={
                        "id": content_block.id,
                        "name": content_block.name,
                        "input": content_block.input
                    }))
                else:
                    # Handle other types if necessary
                    content_blocks.append(ContentBlock(type=content_type, content=content_block.dict()))

            return MessageResponse(conversation_id=conversation_id, content=content_blocks)
        else:
            # Assistant has provided a final answer or other response
            # Return the assistant's response to the client
            for content_block in assistant_response.content:
                content_type = content_block.type
                if content_type == "text":
                    content_blocks.append(ContentBlock(type="text", content={"text": content_block.text}))
                elif content_type == "image":
                    content_blocks.append(ContentBlock(type="image", content={
                        "source": content_block.source.dict()
                    }))
                else:
                    # Handle other types if necessary
                    content_blocks.append(ContentBlock(type=content_type, content=content_block.dict()))

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

Help me finish refactoring. We are trying to remove Gradio GUI and put the logic into a FastAPI endpoint that can be called from a Chrome extension. The goal is to create an API that accepts user messages, processes them through the existing logic (including interactions with the Anthropic API and tools), and returns the assistant's response.

The original code includes tools that interact with the local system (e.g., taking screenshots and clicking). This will be done by the chrome extension now, so we do not need this logic on the server.

In the original logic, screenshots were processed and sent to the assistant, allowing it to make decisions based on the visual state of the user's screen. Since the Chrome extension will now handle the execution of tools (like taking screenshots), we need to adjust our API to accept base64-encoded images from the client and include them in the conversation with the assistant.

To adjust the conversation flow and refactor the code to work as an API endpoint, we need to modify the way the assistant interacts with tool usage. Since the execution of tools (like taking screenshots) will now be handled by the client (Chrome extension), the server-side API should only pass tool definitions to the assistant and manage the conversation state without executing any tools.

2. **Remove Server-Side Tool Execution**:

   - The server should not execute any tools. Instead, it should only provide the tool definitions to the assistant.
   - The `ToolCollection` class should be adjusted to include only the tool definitions without any execution logic.

3. **Maintain Message Format and Conversation State**:

   - Ensure that the messages are formatted correctly to keep the assistant functioning as expected.
   - The server maintains the conversation state per conversation ID, allowing multiple conversations to occur simultaneously.

4. **Implement API Endpoint to Handle Tool Requests and Results**:
   - The API endpoint accepts user messages and content blocks (including images and tool results), processes them through the assistant, and returns the assistant's response.
   - The assistant's response includes `stop_reason`, which indicates whether the assistant wants to use a tool or has completed the response.
   - The client handles `tool_use` content blocks, executes the tools, and sends back `tool_result` content blocks.

- **Conversation State Management**:

  - The server maintains the conversation state using `conversation_states` dictionary, keyed by `conversation_id`.
  - If a new conversation starts, it initializes the state; otherwise, it retrieves the existing state.

- **Processing User Messages**:

  - The API endpoint accepts `MessageRequest`, which includes `content_blocks` sent by the client.
  - These content blocks can be of type `text`, `image`, or `tool_result`.
  - The server converts these blocks into the format expected by the assistant and appends them to the conversation.

- **Assistant Interaction**:

  - The server calls the assistant via the `AnthropicActor`, passing the messages and tool definitions.
  - The assistant processes the messages and may respond with a `stop_reason` of `tool_use` if it wants to use a tool.

- **Handling Assistant Responses**:

  - If the assistant's `stop_reason` is `tool_use`, the server returns the assistant's response to the client without executing any tools.
  - The client's responsibility is to execute the tool (e.g., take a screenshot) and send back the tool results as a new message.
  - If the assistant provides a final answer, the server returns the response to the client.

- **Maintaining Message Format**:

  - The server ensures that the messages are formatted correctly, preserving the content types and structures expected by the assistant.

- **Client Responsibility**:
  - The client (Chrome extension) interprets `tool_use` content blocks, executes the required tools locally, and sends back `tool_result` content blocks.
  - This allows the assistant to continue the conversation based on the tool results.

By following this approach, we adjust the conversation flow to allow the client to execute tools and send back results while keeping the assistant functioning correctly. The server-side API becomes stateless in terms of tool execution and only manages the conversation state and interaction with the assistant.
