Here is the loop again:

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

What i want is to maintain state on the server, and keep re-entering this loop using the conversation id. Here is the class again:


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


Add this functionality to my API endpoint. Currenty, it is not working at all. I get:

```sh
{
  "detail": "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'prompt must start with \"\\n\\nHuman:\" turn after an optional system prompt'}}"
}
```

Here is computer_use.py


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
        state["model"] = "claude-2.1"
    if "system_prompt" not in state:
        device_os_name = (
            "Windows"
            if os.name == "nt"
            else "Mac"
            if os.uname().sysname == "Darwin"
            else "Linux"
        )
        state["system_prompt"] = f"""You are an assistant that can interact with the user's computer via a Chrome extension.
When you need to perform an action on the user's computer (e.g., clicking, typing, taking a screenshot), you should output a JSON command that the Chrome extension can interpret and execute.
Your responses should be in the following format:

{{
    "action": "action_name",
    "parameters": {{
        "param1": "value1",
        ...
    }}
}}
Assistant: The current date is {datetime.today().strftime('%A, %B %d, %Y')}.

<SYSTEM_CAPABILITY>
* You are utilizing a {device_os_name} machine with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>"""
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
        self.provider = provider
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images

        self.tool_collection = ToolCollection(
            ComputerTool(selected_screen=selected_screen)
        )

        self.system = (
            f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
        )

        # Instantiate the appropriate API client based on the provider

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


Here is my Router:


```python
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