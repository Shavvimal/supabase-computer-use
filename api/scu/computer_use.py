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
