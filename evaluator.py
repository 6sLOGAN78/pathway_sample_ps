import os
from dotenv import load_dotenv
from typing import Dict, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()
API_KEY_BACKUP = os.getenv("GOOGLE_API_KEY")
if not API_KEY_BACKUP:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")


# Schema
class MemoryDecision(BaseModel):
    Need_web: str = Field(
        description="Does this query require real-time or factual data?",
        pattern="^(yes|no)$"
    )
    update_user_data: bool = Field(
        description="True if query contains new user information or preferences."
    )
    new_info: Optional[str] = Field(
        default=None,
        description="Summarized or cleaned piece of user data to remember (if any)."
    )


# Prompt 
SYSTEM_PROMPT = SystemMessage(content="""
You are an intelligent evaluator agent.

Your job:
1. Decide if the user query needs *real-time* or *factual* information from the web (set Need_web = "yes" or "no").
2. If the user shares **personal, factual, or preference data** — such as their name, email, account info, interests, likes, dislikes, notes, or facts to remember — then:
   - Set update_user_data = true
   - Extract the *essential fact only* as a short statement (1 sentence max).
   - Write it in `new_info`.

Examples:
User: "i lost my bank account , i love intraday tradin, i love tata steel stocs, my phone number is, remember this" ----------------" "
→ update_user_data = true
→ new_info = ""

User: "What is the current stock price of Tesla?"
→ update_user_data = false
→ Need_web = "yes"

Always respond as **valid JSON only**:
{
  "Need_web": "yes" or "no",
  "update_user_data": true or false,
  "new_info": "..." or null
}
""")

# Decision Agent 
class DecisionAgent:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = API_KEY_BACKUP
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )
        self.structured_model = self.model.with_structured_output(MemoryDecision)

    def analyze_query(self, query: str) -> Dict:
        """Analyze user query and extract decision."""
        parsed_result: MemoryDecision = self.structured_model.invoke([
            SYSTEM_PROMPT,
            HumanMessage(content=query)
        ])
        return {
            "Need_web": parsed_result.Need_web,
            "update_user_data": parsed_result.update_user_data,
            "new_info": parsed_result.new_info
        }
