from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
from api.utils.constants import SHARED
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from typing import Dict, Any
from .computer_use import (
    setup_state,
    MessageRequest,
    MessageResponse,
    ContentBlock,
    Sender
)
from supabase import Client

router = APIRouter()

@router.post("/agent")
async def agent_endpoint(request: MessageRequest):
    conversation_id = request.conversation_id or str(uuid4())
    supabase: Client = SHARED["supabase_client"]

    # Retrieve or create the conversation
    if request.conversation_id:
        # Fetch the conversation from the database
        conversation_response = supabase.table('conversations').select('*').eq('id', conversation_id).execute()
        if not conversation_response.data:
            # Conversation not found
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        # Create a new conversation
        supabase.table('conversations').insert({'id': conversation_id}).execute()

    # Now, fetch the messages for the conversation
    messages_response = supabase.table('messages').select('*').eq('conversation_id', conversation_id).order('created_at').execute()
    messages = messages_response.data  # List of messages

    # Prepare messages for the assistant
    assistant_messages = []
    for message in messages:
        assistant_messages.append({
            "role": message['sender'],
            "content": message['content_blocks'],
        })

    # Convert content_blocks to the format expected by the assistant
    user_content = []
    initial_message_text = ""
    for block in request.content_blocks:
        if block.type == "text":
            text = block.content.get("text", "")
            initial_message_text += text
            user_content.append({
                "type": "text",
                "text": text
            })
        elif block.type == "image":
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": block.content.get("source", {}).get("data", "")
                }
            })
        elif block.type == "tool_result":
            # Handle tool results
            user_content.append({
                "type": "tool_result",
                "tool_use_id": block.content.get("tool_use_id"),
                "is_error": block.content.get("is_error", False),
                "content": block.content.get("content", [])
            })
            # Store tool usage data
            tool_usage_data = {
                "conversation_id": conversation_id,
                "tool_name": "unknown",  # Update this if tool name is available
                "tool_input": {},  # Update this if tool input is available
                "tool_output": block.content.get("content", []),
                "is_error": block.content.get("is_error", False),
            }
            tool_usage_response = supabase.table('tool_usages').insert(tool_usage_data).execute()
        else:
            # Handle other types as needed
            user_content.append(block.dict())

    # If conversation_id is not provided, invoke the extraction agent
    if not request.conversation_id:
        if initial_message_text:
            # Invoke the agentic_rag
            agentic_rag = SHARED["agentic_rag"]
            try:
                result = agentic_rag.invoke(initial_message_text)
                # Replace the user_content with the result from extraction_agent
                user_content = [{
                    "type": "text",
                    "text": result
                }]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error invoking extraction_agent: {str(e)}")
        else:
            # No initial message text provided
            raise HTTPException(status_code=400, detail="No initial message text provided for extraction.")

    # Append the user's message to assistant_messages
    assistant_messages.append({
        "role": Sender.USER,
        "content": user_content,
    })

    # Append the user's content to the conversation
    user_message = {
        "conversation_id": conversation_id,
        "sender": Sender.USER,
        "content_blocks": user_content,
    }
    message_response = supabase.table('messages').insert(user_message).execute()

    try:
        # Access the actor and pass metadata
        anthropic_actor = SHARED["anthropic_actor"]
        assistant_response = anthropic_actor(messages=assistant_messages, metadata=request.metadata)

        # Convert assistant_response.content to serializable format
        assistant_content = []
        for content_block in assistant_response.content:
            content_type = content_block.type
            if content_type == "text":
                assistant_content.append({
                    "type": "text",
                    "text": content_block.text
                })
            elif content_type == "image":
                assistant_content.append({
                    "type": "image",
                    "source": content_block.source.dict()
                })
            elif content_type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": content_block.id,
                    "name": content_block.name,
                    "input": content_block.input
                })
            else:
                # Handle other types as necessary
                assistant_content.append(content_block.dict())

        # Add assistant's message to messages
        assistant_message = {
            "conversation_id": conversation_id,
            "sender": Sender.BOT,
            "content_blocks": assistant_content,
        }

        # Add assistant's message to messages
        assistant_message_response = supabase.table('messages').insert(assistant_message).execute()

        # Process assistant's response content for the client
        content_blocks = []
        for content_block in assistant_response.content:
            content_type = content_block.type
            if content_type == "text":
                content_blocks.append(ContentBlock(type="text", content={"text": content_block.text}))
            elif content_type == "image":
                content_blocks.append(ContentBlock(type="image", content={
                    "source": content_block.source.dict()
                }))
            elif content_type == "tool_use":
                content_blocks.append(ContentBlock(type="tool_use", content={
                    "id": content_block.id,
                    "name": content_block.name,
                    "input": content_block.input
                }))
            else:
                # Handle other types as necessary
                content_blocks.append(ContentBlock(type=content_type, content=content_block.dict()))

        # Return the assistant's response to the client
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
        agent = SHARED["agentic_rag"]
        supabase = SHARED["supabase_client"]
        result = agent.invoke(query)

        # Add new companies to the Supabase
        # background_tasks.add_task(add_apollo_companies_supabase, supabase, companies)

        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
