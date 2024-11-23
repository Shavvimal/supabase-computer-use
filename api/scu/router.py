from uuid import uuid4
from fastapi import HTTPException, Depends, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from api.core.auth import verify_token
from api.utils.constants import SHARED
from typing import Dict, Any, List, Optional

from api.scu.computer_use import (
    setup_state,
    AnthropicActor,
    process_content_block,
    MessageRequest,
    MessageResponse,
)
router = APIRouter()


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


# In-memory store for conversation states
conversation_states: Dict[str, Dict[str, Any]] = {}

# Endpoint definition
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
            "content": [{"type": "text", "text": request.message}],
        }
    )

    try:
        # Create the actor
        actor = AnthropicActor(
            model=state["model"],
            system_prompt_suffix=state["custom_system_prompt"],
            api_key=state["api_key"],
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

        # Process the assistant's response content blocks
        content_blocks = []
        for content_block in response.content:
            processed_block = process_content_block(content_block)
            if processed_block:
                content_blocks.append(processed_block)

        return MessageResponse(conversation_id=conversation_id, content=content_blocks)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
