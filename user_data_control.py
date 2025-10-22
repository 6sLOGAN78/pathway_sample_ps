# user_data_Control.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types
import tiktoken
from dotenv import load_dotenv
load_dotenv()
# ---------------- Config ----------------
USER_MEMORY_PATH = "toy_user_Detail.txt"
TOKEN_LIMIT = 2000

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment variables")

# ---------------- Initialize AI ----------------
enc = tiktoken.encoding_for_model("gpt-4")
genai_client = genai.Client(api_key=API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

# ---------------- Helpers ----------------
def load_user_memory() -> str:
    if not os.path.exists(USER_MEMORY_PATH):
        with open(USER_MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write("User memory initialized.\n")
    with open(USER_MEMORY_PATH, "r", encoding="utf-8") as f:
        return f.read()

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def save_user_memory(text: str):
    with open(USER_MEMORY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[INFO] User memory updated. Current tokens: {count_tokens(text)}")

# ---------------- Summarization ----------------
def summarize_memory(memory_text: str) -> str:
    """
    Calls AI to remove least important points to reduce tokens under TOKEN_LIMIT.
    """
    system_prompt = SystemMessage(content=f"""
You are a finance analytics assistant.
You are given a user's memory.
Remove the least important points and shorten it to be under {TOKEN_LIMIT} tokens.
Keep essential financial preferences, emails, account info, and important notes
and if he likes any stocks trading or financial thing or his major decision.
and remove any irrelevant or less important details.and remove if any things comes more than once.
Return the updated memory as plain text.
""")
    human_prompt = HumanMessage(content=memory_text)

    try:
        result = model.invoke([system_prompt, human_prompt])
        summarized_text = str(result.content)

        # Safety fallback: truncate if AI still exceeds limit
        if count_tokens(summarized_text) > TOKEN_LIMIT:
            tokens = enc.encode(summarized_text)[:TOKEN_LIMIT]
            summarized_text = enc.decode(tokens)

        return summarized_text
    except Exception as e:
        print(f"[ERROR] Failed to summarize memory: {e}")
        # fallback: truncate manually
        tokens = enc.encode(memory_text)[:TOKEN_LIMIT]
        return enc.decode(tokens)

# ---------------- Main ----------------
if __name__ == "__main__":
    user_memory = load_user_memory()
    print(f"[INFO] Current tokens in user memory: {count_tokens(user_memory)}")

    if count_tokens(user_memory) > TOKEN_LIMIT:
        print("[INFO] Token limit exceeded. Summarizing user memory...")
        new_memory = summarize_memory(user_memory)
        save_user_memory(new_memory)
    else:
        print("[INFO] User memory within token limit. No changes made.")
