from fastapi import  BackgroundTasks
from fastapi.responses import JSONResponse
from api.utils.constants import SHARED
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from typing import Dict, Any, List, Optional
from anthropic.types import TextBlock
from typing import Any, cast
from .computer_use import (
    setup_state,
    AnthropicActor,
    MessageRequest,
    MessageResponse,
    ContentBlock,
    Sender
)
from anthropic.types.beta import (
    BetaContentBlockParam,
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

        content_blocks = []

        # Process assistant's response content
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

