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
import base64  # Import base64 module for validation

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
        sanitized_content_blocks = []
        for content_block in message['content_blocks']:
            if content_block['type'] == 'image':
                image_data = content_block.get('source', {}).get('data', '')
                if image_data:
                    try:
                        base64.b64decode(image_data)
                        # Valid base64, include the image block
                        sanitized_content_blocks.append(content_block)
                    except Exception:
                        # Invalid base64 data, skip this block
                        pass
                else:
                    # Empty data, skip this block
                    pass
            else:
                # For other types, include the content block
                sanitized_content_blocks.append(content_block)
        assistant_messages.append({
            "role": message['sender'],
            "content": sanitized_content_blocks,
        })

    # Initialize dictionaries to keep track of tool_uses and tool_results
    tool_use_blocks = {}
    tool_result_blocks = {}

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
            image_data = block.content.get("source", {}).get("data", "")
            if image_data:
                try:
                    base64.b64decode(image_data)
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    })
                except Exception:
                    # Invalid base64 data, skip this block
                    pass
            else:
                # Skip adding image block if data is empty
                pass
        elif block.type == "tool_use":
            # Collect tool_use blocks
            tool_use_id = block.content.get("id")
            tool_use_blocks[tool_use_id] = block
        elif block.type == "tool_result":
            # Collect tool_result blocks
            tool_use_id = block.content.get("tool_use_id")
            tool_result_blocks[tool_use_id] = block
        # else:
            # Handle other types as needed

    # Now, process the tool_result blocks
    for tool_use_id, result_block in tool_result_blocks.items():
        # Check if there's a matching tool_use with this ID
        has_matching_tool_use = tool_use_id in tool_use_blocks
        if not has_matching_tool_use:
            # No matching tool_use, include the tool_result in user_content
            user_content.append({
                "type": "tool_result",
                "tool_use_id": result_block.content.get("tool_use_id"),
                "is_error": result_block.content.get("is_error", False),
                "content": result_block.content.get("content", [])
            })
            # Store tool usage data
            tool_usage_data = {
                "conversation_id": conversation_id,
                "tool_name": "unknown",  # Update if tool name is available
                "tool_input": {},  # Update if tool input is available
                "tool_output": result_block.content.get("content", []),
                "is_error": result_block.content.get("is_error", False),
            }
            supabase.table('tool_usages').insert(tool_usage_data).execute()

    # If conversation_id is not provided, invoke the extraction agent
    if not request.conversation_id:
        if initial_message_text:
            # Invoke the agentic_rag
            agentic_rag = SHARED["agentic_rag"]
            try:
                result = agentic_rag.invoke(initial_message_text)
                print(result)
                # Replace just the first text content with the result
                if len(user_content) > 0:
                    user_content[0] = {
                        "type": "text",
                        "text": result
                    }
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

    # Store the user's message in the database
    user_message = {
        "conversation_id": conversation_id,
        "sender": Sender.USER,
        "content_blocks": user_content,
    }
    supabase.table('messages').insert(user_message).execute()

    # Process tool_use blocks and append corresponding messages
    for tool_use_id, tool_use_block in tool_use_blocks.items():
        matching_result = tool_result_blocks.get(tool_use_id)
        if matching_result:
            # Append tool_use as assistant message
            tool_use_content = [{
                "type": "tool_use",
                "id": tool_use_block.content.get("id"),
                "name": tool_use_block.content.get("name"),
                "input": tool_use_block.content.get("input", {})
            }]
            assistant_messages.append({
                "role": Sender.BOT,
                "content": tool_use_content,
            })
            # Store assistant's tool_use message
            assistant_message = {
                "conversation_id": conversation_id,
                "sender": Sender.BOT,
                "content_blocks": tool_use_content,
            }
            supabase.table('messages').insert(assistant_message).execute()

            # Append tool_result as user message
            tool_result_content = [{
                "type": "tool_result",
                "tool_use_id": matching_result.content.get("tool_use_id"),
                "is_error": matching_result.content.get("is_error", False),
                "content": matching_result.content.get("content", [])
            }]
            assistant_messages.append({
                "role": Sender.USER,
                "content": tool_result_content,
            })
            # Store user's tool_result message
            user_message = {
                "conversation_id": conversation_id,
                "sender": Sender.USER,
                "content_blocks": tool_result_content,
            }
            supabase.table('messages').insert(user_message).execute()

            # Store tool usage data
            tool_usage_data = {
                "conversation_id": conversation_id,
                "tool_name": tool_use_block.content.get("name", "unknown"),
                "tool_input": tool_use_block.content.get("input", {}),
                "tool_output": matching_result.content.get("content", []),
                "is_error": matching_result.content.get("is_error", False),
            }
            supabase.table('tool_usages').insert(tool_usage_data).execute()

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
                # Validate image data
                image_data = content_block.source.data
                if image_data:
                    try:
                        base64.b64decode(image_data)
                        assistant_content.append({
                            "type": "image",
                            "source": content_block.source.dict()
                        })
                    except Exception:
                        # Invalid base64 data, skip this block
                        pass
                else:
                    # Skip if data is empty
                    pass
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

        # Store assistant's message in the database
        assistant_message = {
            "conversation_id": conversation_id,
            "sender": Sender.BOT,
            "content_blocks": assistant_content,
        }
        supabase.table('messages').insert(assistant_message).execute()

        # Prepare response for the client
        content_blocks = []
        for content_block in assistant_response.content:
            content_type = content_block.type
            if content_type == "text":
                content_blocks.append(ContentBlock(type="text", content={"text": content_block.text}))
            elif content_type == "image":
                image_data = content_block.source.data
                if image_data:
                    try:
                        base64.b64decode(image_data)
                        content_blocks.append(ContentBlock(type="image", content={
                            "source": content_block.source.dict()
                        }))
                    except Exception:
                        # Invalid base64 data, skip this block
                        pass
                else:
                    # Skip if data is empty
                    pass
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
