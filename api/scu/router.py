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

    # Handle user message or content_blocks
    if request.message:
        # User sent a text message
        user_content = [TextBlock(type="text", text=request.message)]
    elif request.content_blocks:
        # User sent content blocks (e.g., tool results)
        user_content = request.content_blocks
    else:
        raise HTTPException(status_code=400, detail="No message or content_blocks provided.")


    # Append the user's message to the conversation
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
            "content": cast(List[BetaContentBlockParam], assistant_response.content),
        })

        content_blocks = []

        # Process assistant's response content
        for content_block in assistant_response.content:
            content_type = content_block.type
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

