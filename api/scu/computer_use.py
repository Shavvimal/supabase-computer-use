import os
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any

from fastapi import HTTPException
from pydantic import BaseModel
from anthropic import Anthropic, APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaTextBlock


# Pydantic models for request and response
class MessageRequest(BaseModel):
    conversation_id: str = None
    message: str

class MessageResponse(BaseModel):
    conversation_id: str
    response: str

# In-memory store for conversation states
conversation_states: Dict[str, Dict[str, Any]] = {}

# Helper functions
def setup_state(state):
    if "messages" not in state:
        state["messages"] = []
    if "api_key" not in state:
        state["api_key"] = os.getenv("ANTHROPIC_API_KEY", "")
        if not state["api_key"]:
            raise ValueError("API key not found. Please set it in the environment.")
    if "model" not in state:
        state["model"] = "claude-2"
    if "custom_system_prompt" not in state:
        state["custom_system_prompt"] = (
            f"\n\nNOTE: You are a helpful assistant. The current date is "
            f"{datetime.today().strftime('%A, %B %d, %Y')}."
        )

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _render_message(message, state):
    if isinstance(message, BetaTextBlock):
        return message.text
    elif isinstance(message, TextBlock):
        return message.text
    else:
        return str(message)

# Endpoint definition
@app.post("/agent")
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
            "content": [{"type": "text", "text": request.message}],
        }
    )

    try:
        response_state = {}

        # Create the actor
        actor = AnthropicActor(
            model=state["model"],
            system_prompt_suffix=state["custom_system_prompt"],
            api_key=state["api_key"],
            api_response_callback=lambda response: _api_response_callback(response, response_state),
            max_tokens=4096,
        )

        response = actor(messages=state["messages"])

        # Add assistant's message to messages
        state["messages"].append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        # Process the assistant's response
        response_messages = []
        for content_block in response.content:
            rendered_message = _render_message(content_block, state)
            if rendered_message:
                response_messages.append(rendered_message)

        # The assistant's response is the concatenation of rendered messages
        assistant_response = "\n".join(response_messages)

        return MessageResponse(conversation_id=conversation_id, response=assistant_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AnthropicActor class
class AnthropicActor:
    def __init__(
        self,
        model: str,
        system_prompt_suffix: str,
        api_key: str,
        api_response_callback,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.system_prompt_suffix = system_prompt_suffix
        self.api_key = api_key
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens

        # System prompt without local system capabilities
        self.system = f"NOTE: You are a helpful assistant.{system_prompt_suffix}"

        # Instantiate the Anthropic API client
        self.client = Anthropic(api_key=api_key)

    def __call__(
        self,
        *,
        messages: list[BetaMessageParam]
    ):
        # Call the API synchronously
        raw_response = self.client.beta.messages.with_raw_response.create(
            max_tokens=self.max_tokens,
            messages=messages,
            model=self.model,
            system=self.system,
        )

        self.api_response_callback(raw_response)

        response = raw_response.parse()

        return response
