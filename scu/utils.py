import os
from supabase import create_client, Client

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define SessionContext class
class SessionContext:
    def __init__(self, user_id):
        self.user_id = user_id
        self.query_id = None
        self.solution_id = None

# Log Query
def log_query(user_id, query_text, query_embedding, query_type):
    data = {
        "user_id": user_id,
        "query_text": query_text,
        "query_embedding": query_embedding,  # Ensure this is a list or appropriate format
        "type": query_type
    }
    response = supabase.table("queries").insert(data).execute()
    query_id = response.data[0]["id"]
    return query_id

# Log Solution
def log_solution(query_id, feedback, conversation_data):
    data = {
        "query_id": query_id,
        "feedback": feedback,
        "conversation_data": conversation_data
    }
    response = supabase.table("solutions").insert(data).execute()
    solution_id = response.data[0]["id"]
    return solution_id

# Log Conversation Message
def log_conversation_message(solution_id, sender, message_content):
    data = {
        "solution_id": solution_id,
        "sender": sender,
        "message_content": str(message_content)
    }
    supabase.table("conversation_messages").insert(data).execute()

# Log Tool Usage
def log_tool_usage(solution_id, tool_name, tool_input, tool_output):
    data = {
        "solution_id": solution_id,
        "tool_name": tool_name,
        "tool_input": tool_input,   # Ensure it's JSON serializable
        "tool_output": {
            "output": tool_output.output,
            "error": tool_output.error,
            "base64_image": tool_output.base64_image
        }
    }
    supabase.table("tools_used").insert(data).execute()

# Log Screenshot
def log_screenshot(solution_id, uri):
    data = {
        "solution_id": solution_id,
        "uri": uri
    }
    supabase.table("screenshots").insert(data).execute()

# Extract text from content
def extract_text_from_content(content):
    if isinstance(content, list):
        texts = []
        for block in content:
            if block["type"] == "text":
                texts.append(block["text"])
        return "\n".join(texts)
    elif isinstance(content, str):
        return content
    else:
        return ""

# Save Screenshot to Storage
def save_screenshot_to_storage(base64_image_data):
    import base64
    from datetime import datetime
    file_name = f"screenshots/{datetime.now().isoformat()}.png"
    supabase.storage.from_("screenshots").upload(file_name, base64.b64decode(base64_image_data))

    # Get the public URL
    uri = supabase.storage.from_("screenshots").get_public_url(file_name)
    return uri

# Generate Embedding (Placeholder)
def generate_embedding(text):
    # Implement your embedding generation logic here
    # For example, using OpenAI's embedding API
    return [0.0] * 1536  # Replace with actual embedding

# Determine Query Type (Placeholder)
def determine_query_type(text):
    # Implement logic to determine the type of query
    return 1  # Example: return 1 for one type, 2 for another