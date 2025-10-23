import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY_BACKUP = os.getenv("GOOGLE_API_KEY")
if not API_KEY_BACKUP:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

class NamingAgent:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = API_KEY_BACKUP
        self.client = genai.Client(api_key=api_key)

    def generate_name(self, user_messages):
        """
        Takes a list of user messages (strings).
        Uses all messages up to 5 to decide a short meaningful name.
        """
        if not user_messages:
            return "Unnamed Conversation"

        # Consider all messages provided
        combined_text = "\n".join([f"- {msg}" for msg in user_messages[:5]])
        prompt = f"""
You are a conversation naming agent.
Here are the main user messages from a chat session:
{combined_text}

Generate a short, meaningful title (2–5 words) summarizing this conversation.
Only output the name, no explanations.
"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            name = response.text.strip().replace("\n", " ")
            name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            return name or "Unnamed_Conversation"
        except Exception as e:
            print("[ERROR] Failed to generate name:", e)
            return "Unnamed_Conversation"
