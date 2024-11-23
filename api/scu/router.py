from fastapi import  BackgroundTasks
from fastapi.responses import JSONResponse
from api.utils.constants import SHARED
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from supabase import Client
from .computer_use import (
    MessageRequest,
    MessageResponse,
    ContentBlock,
    Sender
)

router = APIRouter()

@router.post("/agent")
async def agent_endpoint(request: MessageRequest):
    conversation_id = request.conversation_id or str(uuid4())

    supabase: Client = SHARED["supabase_client"]

    # Retrieve or create the conversation
    if request.conversation_id:
        # Fetch the conversation from the database
        conversation_response = supabase.table('conversations').select('*').eq('id', conversation_id).execute()
        if conversation_response.error:
            raise HTTPException(status_code=500, detail=f"Error fetching conversation: {conversation_response.error.message}")
        if not conversation_response.data:
            # Conversation not found
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        # Create a new conversation
        conversation_response = supabase.table('conversations').insert({'id': conversation_id}).execute()
        if conversation_response.error:
            raise HTTPException(status_code=500, detail=f"Error creating conversation: {conversation_response.error.message}")

    # Now, fetch the messages for the conversation
    messages_response = supabase.table('messages').select('*').eq('conversation_id', conversation_id).order('created_at').execute()
    if messages_response.error:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {messages_response.error.message}")
    messages = messages_response.data  # List of messages

    # Prepare messages for the assistant
    assistant_messages = []
    for message in messages:
        assistant_messages.append({
            "role": message['sender'],
            "content": message['content_blocks'],
        })

    # Extract the initial message text and convert content_blocks
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
            # Handle images: store them in the storage bucket
            image_data = block.content.get("data", "")
            image_bytes = base64.b64decode(image_data)
            # Generate a unique path for the image
            image_path = f"screenshots/{conversation_id}/{uuid4()}.png"
            # Upload to Supabase storage
            storage_response = supabase.storage.from_('screenshots').upload(image_path, image_bytes)
            if storage_response.error:
                raise HTTPException(status_code=500, detail=f"Error uploading image: {storage_response.error.message}")
            # Store the image path in the content block
            user_content.append({
                "type": "image",
                "source": {
                    "type": "url",
                    "url": f"https://your-supabase-url/storage/v1/object/public/{image_path}"
                }
            })
            # Also, store screenshot metadata
            screenshot_metadata = {
                "conversation_id": conversation_id,
                "screenshot_path": image_path,
                "metadata": {},  # Add any metadata if needed
            }
            screenshot_response = supabase.table('screenshots').insert(screenshot_metadata).execute()
            if screenshot_response.error:
                raise HTTPException(status_code=500, detail=f"Error storing screenshot metadata: {screenshot_response.error.message}")
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
                "tool_input": {},        # Update this if tool input is available
                "tool_output": block.content.get("content", []),
                "is_error": block.content.get("is_error", False),
            }
            tool_usage_response = supabase.table('tool_usages').insert(tool_usage_data).execute()
            if tool_usage_response.error:
                raise HTTPException(status_code=500, detail=f"Error storing tool usage: {tool_usage_response.error.message}")
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

    # Append the user's content to the conversation
    user_message = {
        "conversation_id": conversation_id,
        "sender": Sender.USER,
        "content_blocks": user_content,
    }
    message_response = supabase.table('messages').insert(user_message).execute()
    if message_response.error:
        raise HTTPException(status_code=500, detail=f"Error storing user message: {message_response.error.message}")

    # Now, include the user's message in the assistant_messages
    assistant_messages.append({
        "role": Sender.USER,
        "content": user_content,
    })

    try:
        # Access the actor
        anthropic_actor = SHARED["anthropic_actor"]
        assistant_response = anthropic_actor(messages=assistant_messages)

        # Add assistant's message to messages
        assistant_message = {
            "conversation_id": conversation_id,
            "sender": Sender.BOT,
            "content_blocks": assistant_response.content,
        }
        assistant_message_response = supabase.table('messages').insert(assistant_message).execute()
        if assistant_message_response.error:
            raise HTTPException(status_code=500, detail=f"Error storing assistant message: {assistant_message_response.error.message}")

        # Process assistant's response content
        content_blocks = []

        # Check the assistant's stop_reason
        if assistant_response.stop_reason == 'tool_use':
            # Assistant is requesting tool usage
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
                    # Store tool usage data
                    tool_usage_data = {
                        "conversation_id": conversation_id,
                        "tool_name": content_block.name,
                        "tool_input": content_block.input,
                        "tool_output": None,  # Output not yet available
                        "is_error": False,
                    }
                    tool_usage_response = supabase.table('tool_usages').insert(tool_usage_data).execute()
                    if tool_usage_response.error:
                        raise HTTPException(status_code=500, detail=f"Error storing tool usage: {tool_usage_response.error.message}")
                else:
                    content_blocks.append(ContentBlock(type=content_type, content=content_block.dict()))

            return MessageResponse(conversation_id=conversation_id, content=content_blocks)
        else:
            # Assistant has provided a final answer
            for content_block in assistant_response.content:
                content_type = content_block.type
                if content_type == "text":
                    content_blocks.append(ContentBlock(type="text", content={"text": content_block.text}))
                elif content_type == "image":
                    # Handle images returned by the assistant
                    image_source = content_block.source
                    if image_source.type == "base64":
                        image_data = image_source.data
                        image_bytes = base64.b64decode(image_data)
                        image_path = f"screenshots/{conversation_id}/{uuid4()}.png"
                        storage_response = supabase.storage.from_('screenshots').upload(image_path, image_bytes)
                        if storage_response.error:
                            raise HTTPException(status_code=500, detail=f"Error uploading image: {storage_response.error.message}")
                        content_blocks.append(ContentBlock(type="image", content={
                            "source": {
                                "type": "url",
                                "url": f"https://your-supabase-url/storage/v1/object/public/{image_path}"
                            }
                        }))
                        # Store screenshot metadata
                        screenshot_metadata = {
                            "conversation_id": conversation_id,
                            "screenshot_path": image_path,
                            "metadata": {},  # Add any metadata if needed
                        }
                        screenshot_response = supabase.table('screenshots').insert(screenshot_metadata).execute()
                        if screenshot_response.error:
                            raise HTTPException(status_code=500, detail=f"Error storing screenshot metadata: {screenshot_response.error.message}")
                    else:
                        content_blocks.append(ContentBlock(type="image", content={"source": image_source.dict()}))
                else:
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
        agent = SHARED["agentic_rag"]
        supabase = SHARED["supabase_client"]
        result = agent.invoke(query)

        # Add new companies to the Supabase
        # background_tasks.add_task(add_apollo_companies_supabase, supabase, companies)

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

