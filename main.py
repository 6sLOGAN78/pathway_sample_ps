import os
from datetime import datetime
from dotenv import load_dotenv
import uuid
import tiktoken
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from evaluator import DecisionAgent
from naming_agent import NamingAgent
from web_search import webQuery
from google import genai
from google.genai import types

# Environment 
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

#  Setup Models 
enc = tiktoken.encoding_for_model("gpt-4")
TOKEN_LIMIT = 100000

decision_agent = DecisionAgent(api_key=API_KEY)
web_agent = webQuery(api_key=API_KEY)
naming_agent = NamingAgent(api_key=API_KEY)
genai_client = genai.Client(api_key=API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

# Helper Functions 
def count_tokens(messages):
    return sum(len(enc.encode(msg.content)) for msg in messages)

def list_conversations(folder="conversation"):
    os.makedirs(folder, exist_ok=True)
    return [f for f in os.listdir(folder) if f.endswith(".json")]

def load_conversation_json(path):
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_conversation_json(conversation_data, filename, folder="conversation"):
    import json
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    # Append to existing chat if file exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
        existing_data["chat"].extend(conversation_data["chat"])
        existing_data["documents"].extend([
            doc for doc in conversation_data["documents"]
            if doc not in existing_data.get("documents", [])
        ])
        conversation_data = existing_data

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Conversation saved/updated to {file_path}")
    return file_path

#  User Memory File 
USER_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "user_details.txt")
os.makedirs(os.path.dirname(USER_MEMORY_PATH) or ".", exist_ok=True)
if not os.path.exists(USER_MEMORY_PATH):
    with open(USER_MEMORY_PATH, "w", encoding="utf-8") as f:
        f.write("User memory initialized.\n")

def load_user_memory():
    with open(USER_MEMORY_PATH, "r", encoding="utf-8") as f:
        return f.read()

user_memory = load_user_memory()

def update_user_memory(new_info: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(USER_MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] {new_info}")
    print(f"[INFO] Added to user memory: {new_info}")
    # Reload memory so system prompt sees latest info
    return load_user_memory()

#  System Prompt 
def create_system_prompt():
    return SystemMessage(content=f"""
You are a highly knowledgeable financial advisor.
Answer clearly and professionally.

User memory (from user_details.txt):
{user_memory}

You can use the above user memory to assist the user in their queries.
""")

system_prompt = create_system_prompt()

#  Load Previous Conversation 
conversation_folder = "conversation"
existing_convos = list_conversations(conversation_folder)
conversation_path = None

if existing_convos:
    print("Existing conversations:")
    for f in existing_convos:
        print("-", f)
    choice = input("Do you want to load a previous conversation? (yes/no): ").strip().lower()
    if choice in ("yes", "y"):
        selected = input("Enter the filename of the conversation to load: ").strip()
        if selected in existing_convos:
            conversation_path = os.path.join(conversation_folder, selected)
        else:
            print("[ERROR] Selected file does not exist. Starting new conversation.")
else:
    print("[INFO] No previous conversations found. Starting new conversation.")

#Initialize Conversation 
chat_history = [system_prompt]
uploaded_docs_bytes = []
conversation_id = str(uuid.uuid4())
conversation_name = None
user_messages = []

conversation_data = {
    "conversation_id": conversation_id,
    "conversation_name": "Unnamed Conversation",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "chat": [],
    "documents": []
}

# Load previous conversation if selected
if conversation_path and os.path.exists(conversation_path):
    data = load_conversation_json(conversation_path)
    conversation_id = data["conversation_id"]
    conversation_name = data.get("conversation_name", "Unnamed Conversation")
    conversation_data.update({
        "conversation_id": conversation_id,
        "conversation_name": conversation_name,
        "created_at": data.get("created_at", conversation_data["created_at"])
    })
    for msg in data["chat"]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
            user_messages.append(msg["content"])
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    uploaded_docs_bytes = data.get("documents", [])

# Loop 
while True:
    user_input = input("You (or type 'upload <file_path>' / 'exit'): ").strip()
    if user_input.lower() == "exit":
        print("[INFO] Ending session...")
        break

    # Handle document upload
    if user_input.lower().startswith("upload "):
        file_path = user_input[7:].strip()
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            continue
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            mime_type = "application/pdf" if file_path.lower().endswith(".pdf") else "image/png"
            uploaded_docs_bytes.append({
                "filename": os.path.basename(file_path),
                "content": file_bytes,
                "mime_type": mime_type,
                "path": file_path
            })
            conversation_data["documents"].append({
                "filename": os.path.basename(file_path),
                "path": file_path
            })
            print(f"[INFO] Uploaded document: {file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to read file: {e}")
        continue

    # Step 1: Evaluate query
    decision = decision_agent.analyze_query(user_input)
    need_web = decision["Need_web"] == "yes"
    update_user = decision["update_user_data"]
    new_info = decision.get("new_info")

    # Step 2: Web search 
    web_context = ""
    web_sources = []
    if need_web:
        try:
            web_result = web_agent.query(user_input)
            web_context = web_result["text"]
            web_sources = web_result["sources"]
        except Exception as e:
            print(f"[ERROR] Web query failed: {e}")

    # --- Step 3: Check token limit ---
    messages_to_send = chat_history[:] + [HumanMessage(content=user_input)]
    if count_tokens(messages_to_send) > TOKEN_LIMIT:
        print("[ERROR] Token limit exceeded.")
        break

    # Step 4: Process uploaded docs 
    doc_summaries = []
    for doc in uploaded_docs_bytes:
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=doc["content"], mime_type=doc["mime_type"]),
                    f"Give a detailed description of this document: {doc['filename']}"
                ]
            )
            summary_text = response.text
            doc_summaries.append(f"[Summary of {doc['filename']}]: {summary_text}")
        except Exception as e:
            print(f"[ERROR] Failed to summarize {doc['filename']}: {e}")

    if doc_summaries:
        messages_to_send.append(HumanMessage(content="\n".join(doc_summaries)))
    if web_context:
        messages_to_send.append(HumanMessage(content=f"[Web Search Results]: {web_context}\nSources: {', '.join(web_sources)}"))


    try:
        result = model.invoke(messages_to_send)
        ai_response = str(result.content)
    except Exception as e:
        print(f"[ERROR] Failed to generate AI response: {e}")
        continue

    #  Update chat history 
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=ai_response))
    user_messages.append(user_input)
    conversation_data["chat"].append({"role": "user", "content": user_input})
    conversation_data["chat"].append({"role": "assistant", "content": ai_response})

    # Print output 
    print("\nAI:", ai_response)
    if web_sources:
        print("\nWeb Sources:")
        for src in web_sources:
            print("-", src)

    # Dynamic Name Generation 
    if conversation_data["conversation_name"] == "Unnamed Conversation":
        name = naming_agent.generate_name(user_messages)
        conversation_data["conversation_name"] = name
        print(f"[INFO] Conversation named: {name}")

    # Append new info to user_details.txt 
    if update_user and new_info:
        user_memory = update_user_memory(new_info)
        # Update system prompt so AI sees latest memory immediately
        chat_history[0] = create_system_prompt()

#  Save Uploaded Docs 
if uploaded_docs_bytes:
    save_choice = input("You uploaded documents during this session. Do you want to save them for future use? (yes/no): ").strip().lower()
    if save_choice in ("yes", "y"):
        from document_saver import save_documents_for_future
        save_documents_for_future(uploaded_docs_bytes)
        print("[INFO] Documents have been saved and embedded into ChromaDB.")

# Save JSON Conversation 
filename = f"{conversation_data['conversation_name'].replace(' ', '_')}.json"
save_conversation_json(conversation_data, filename, folder="conversation")
print("[INFO] Session complete. Conversation saved.")
