import os
from dotenv import load_dotenv
from web_search import webQuery
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class UserPreferencesAgent:
    def __init__(self, user_details_path="user_details.txt", api_key=None):
        self.user_details_path = user_details_path
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GOOGLE_API_KEY")

        # Web search agent
        self.web_agent = webQuery(api_key=api_key)

        # Chat model for generating search query
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )

    def _read_user_details(self):
        if not os.path.exists(self.user_details_path):
            return ""
        with open(self.user_details_path, "r", encoding="utf-8") as f:
            return f.read()

    def _generate_search_query(self, user_text: str) -> str:
        """
        Use ChatGoogleGenerativeAI to generate a search query from user likes/dislikes.
        """
        system_prompt = f"""
        You are a financial analyst AI.
        Analyze the following user financial preferences.
        Identify their most important likes and dislikes regarding stocks, markets, sectors, and investments.
        Generate a concise, actionable web search query that can be used to fetch news or updates relevant to these preferences
        the web query should ask for current and interesting news according to user details don't put unnecessary text and anything just give web query ."
        
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        response = self.model.invoke(messages)
        return str(response.content).strip()

    def fetch_news_based_on_preferences(self) -> dict:
        user_text = self._read_user_details()
        if not user_text.strip():
            return {"text": "No user details found.", "sources": []}

        # Step 1: Generate search query
        search_query = self._generate_search_query(user_text)

        # Step 2: Query the web
        result = self.web_agent.query(search_query)
        return result


# --- Example usage ---
if __name__ == "__main__":
    agent = UserPreferencesAgent()
    news = agent.fetch_news_based_on_preferences()
    print("AI-generated query results / Summary:", news["text"])
    print("Sources:", news["sources"])
