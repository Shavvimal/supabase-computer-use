import os
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, Depends, APIRouter
from pydantic import BaseModel
from api.core.auth import verify_token
from anthropic import Anthropic, APIResponse
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
    BetaContentBlock,
    BetaTextBlock,
    BetaToolUseBlock,
)

router = APIRouter(dependencies=[Depends(verify_token)])




# Helper functions
def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "api_key" not in state:
        state["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
        if not state["api_key"]:
            raise ValueError("API key not found. Please set it in the environment.")
    if "model" not in state:
        state["model"] = "claude-3-5-sonnet-20241022"
    if "custom_system_prompt" not in state:
        device_os_name = (
            "Windows"
            if os.name == "nt"
            else "Mac"
            if os.uname().sysname == "Darwin"
            else "Linux"
        )
        state["custom_system_prompt"] = f"\n\nNOTE: you are operating a {device_os_name} machine. The current date is {datetime.today().strftime('%A, %B %d, %Y')}."
    if "responses" not in state:
        state["responses"] = {}
    if "tools" not in state:
        state["tools"] = {}
    if "only_n_most_recent_images" not in state:
        state["only_n_most_recent_images"] = 2  # 10

# Pydantic models for request and response
class MessageRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    name: Optional[str] = None
    input: Optional[dict] = None
    id: Optional[str] = None

class MessageResponse(BaseModel):
    conversation_id: str
    content: List[ContentBlock]


def process_content_block(content_block: BetaContentBlock) -> Optional[ContentBlock]:
    if isinstance(content_block, BetaTextBlock):
        return ContentBlock(
            type='text',
            text=content_block.text
        )
    elif isinstance(content_block, BetaToolUseBlock):
        return ContentBlock(
            type='tool_use',
            name=content_block.name,
            input=content_block.input,
            id=content_block.id
        )
    else:
        # Handle other types if necessary
        return None

# AnthropicActor class
class AnthropicActor:
    def __init__(
        self,
        model: str,
        system_prompt_suffix: str,
        api_key: str,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
        self.max_tokens = max_tokens

        # System prompt with system capabilities
        self.system = f"<SYSTEM_CAPABILITY>\n* You are utilizing a machine with internet access.\n* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n</SYSTEM_CAPABILITY>{' ' + system_prompt_suffix if system_prompt_suffix else ''}"

        # Instantiate the Anthropic API client
        self.client = Anthropic(api_key=api_key)

        # Define the tools (only include names and types; execution will be on the client)
        self.tools = [
            {
                "name": "computer",
                "type": "computer_20241022",
                "display_width_px": 1024,
                "display_height_px": 768,
                "display_number": None,
            },
            {
                "name": "bash",
                "type": "bash_20241022",
            },
            {
                "name": "str_replace_editor",
                "type": "text_editor_20241022",
            },
            # Add other tools if necessary
        ]

    def __call__(
        self,
        *,
        messages: List[BetaMessageParam]
    ) -> BetaMessage:
        # Call the API synchronously
        response = self.client.beta.messages.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=self.system,
            tools=self.tools,
            betas=["computer-use-2024-10-22"],
        )

        return response
