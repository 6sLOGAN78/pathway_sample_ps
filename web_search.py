import os
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()

class webQuery:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not provided and GOOGLE_API_KEY not set in environment.")
        self.client = Client(api_key=api_key)

    def query(self, user_query: str) -> dict:
        """
        Returns a dictionary with:
        - 'text': AI answer
        - 'sources': list of website titles or "title (URL)" if URL exists
        """
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_query,
            config={"tools": [{"google_search": {}}]}
        )

        sources = []
        try:
            chunks = response.candidates[0].grounding_metadata.grounding_chunks
            for chunk in chunks:
                if hasattr(chunk, "web"):
                    title = getattr(chunk.web, "title", None)
                    url = getattr(chunk.web, "url", None)
                    if title:
                        if url:
                            sources.append(f"{title} ({url})")
                        else:
                            sources.append(title)
        except Exception:
            sources = []

        return {"text": response.text, "sources": sources}


# # --- Example usage ---
# if __name__ == "__main__":
#     search = webQuery()
#     result = search.query("Current price of Bitcoin")
#     print("AI Answer:", result["text"])
#     print("Sources:", result["sources"])
