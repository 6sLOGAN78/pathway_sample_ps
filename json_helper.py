import os
import json
import datetime
import uuid

CONVERSATIONS_DIR = "conversations"

# Ensure folder exists
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

def create_conversation() -> dict:
    """
    Create a new conversation with unique ID.
    """
    conversation_id = str(uuid.uuid4())
    conversation = {
        "conversation_id": conversation_id,
        "conversation_name": None,  
        "created_at": str(datetime.datetime.utcnow()),
        "messages": [],
        "documents": []
    }
    return conversation

def save_conversation(conversation: dict):
    """
    Save conversation as JSON file.
    Filename = conversation_name.json (if exists) or conversation_id.json
    """
    name = conversation["conversation_name"]
    filename = f"{name.replace(' ', '_')}.json" if name else f"{conversation['conversation_id']}.json"
    path = os.path.join(CONVERSATIONS_DIR, filename)
    with open(path, "w") as f:
        json.dump(conversation, f, indent=2)
    return path

def load_conversation(filename: str) -> dict:
    """
    Load conversation from a JSON file.
    """
    path = os.path.join(CONVERSATIONS_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

def list_conversations() -> list:
    """
    List all JSON conversation files.
    """
    files = [f for f in os.listdir(CONVERSATIONS_DIR) if f.endswith(".json")]
    conversations = []
    for f in files:
        data = load_conversation(f)
        conversations.append({
            "filename": f,
            "conversation_id": data.get("conversation_id"),
            "conversation_name": data.get("conversation_name"),
            "created_at": data.get("created_at")
        })
    return conversations
