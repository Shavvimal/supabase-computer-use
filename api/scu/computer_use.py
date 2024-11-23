import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

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

        # System prompt with system capabilities and instructions
        self.system = f"""
You are an assistant that can interact with the user's computer via a Chrome extension.
When you need to perform an action on the user's computer (e.g., clicking, typing, taking a screenshot), you should output a JSON command that the Chrome extension can interpret and execute.
Your responses should be in the following format:

{{
    "action": "action_name",
    "parameters": {{
        "param1": "value1",
        ...
    }}
}}

For example, to move the mouse to coordinates (100, 200), you would output:

{{
    "action": "move_mouse",
    "parameters": {{
        "x": 100,
        "y": 200
    }}
}}

Do not include any other text in your response when outputting a command.
If you are just replying to the user, respond normally.

Examples:

User: Open a new browser tab.

Assistant:
{{
    "action": "open_tab",
    "parameters": {{
        "url": "about:newtab"
    }}
}}

User: Type "Hello, world!"

Assistant:
{{
    "action": "type_text",
    "parameters": {{
        "text": "Hello, world!"
    }}
}}

User: What is the current date?

Assistant: The current date is {datetime.today().strftime('%A, %B %d, %Y')}.

<SYSTEM_CAPABILITY>
* You are utilizing a machine with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
{self.system_prompt_suffix}
"""

        # Instantiate the Anthropic API client
        self.client = Anthropic(api_key=api_key)

    def __call__(
        self,
        *,
        messages: List[dict],
    ) -> str:
        # Build the prompt
        full_prompt = self.system + "\n\n"

        # Append messages to the prompt
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                full_prompt += f"{HUMAN_PROMPT} {content}\n"
            elif role == "assistant":
                full_prompt += f"{AI_PROMPT} {content}\n"
        full_prompt += AI_PROMPT

        # Call the API synchronously
        response = self.client.completions.create(
            prompt=full_prompt,
            model=self.model,
            max_tokens_to_sample=self.max_tokens,
            stop_sequences=[HUMAN_PROMPT],
        )
        return response.completion.strip()
