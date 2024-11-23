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
    metadata: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    conversation_id: str
    content: List[ContentBlock]

class AnthropicActor:
    def __init__(
        self,
        max_tokens: int = 4096,
    ):
        self.model = "claude-3-5-sonnet-20241022"
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or ""
        self.max_tokens = max_tokens
        # Instantiate the Anthropic API client
        self.client = Anthropic(api_key=self.api_key)
        # Pass tools to the assistant but we won't execute them on the server
        self.tool_collection = ToolCollection(
            ComputerTool(),
        )

    def __call__(
        self,
        *,
        messages: List[BetaMessageParam],
        metadata: Optional[Dict[str, Any]] = None  # Accept metadata
    ):
        # Generate system prompt with metadata
        system = self.generate_system_prompt(metadata)


        # Call the API synchronously
        raw_response = self.client.beta.messages.with_raw_response.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=system,
            tools=self.tool_collection.to_params(),
            betas=["computer-use-2024-10-22"],
        )

        response = raw_response.parse()

        return response

    def generate_system_prompt(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        # Build the system prompt using the metadata
        system_prompt = f"""<SYSTEM_CAPABILITY>
        * You are operating within a Chrome browser environment with internet access.
        * The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
        * You can interact with the browser and web pages directly using the provided tools.
        * You should not attempt to open another browser; interact with the web directly through the current browser.
        </SYSTEM_CAPABILITY>
        """
        if metadata:
            # Incorporate metadata into the system prompt
            browser_info = metadata.get('browser', {})
            user_agent = browser_info.get('userAgent', '')
            system_prompt += f"\n<SYSTEM_METADATA>\n"
            system_prompt += f"* Browser User Agent: {user_agent}\n"
            system_prompt += f"* Screen Info: {metadata.get('screen', {})}\n"
            system_prompt += f"* Window Info: {metadata.get('window', {})}\n"
            system_prompt += f"* Connection Info: {metadata.get('connection', {})}\n"
            system_prompt += "</SYSTEM_METADATA>\n"
        return system_prompt

