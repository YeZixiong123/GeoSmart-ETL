import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataInsightAgent")

class DataInsightAgent:
    def __init__(self):
        self.api_key = os.getenv("AI_API_KEY")
        self.base_url = os.getenv("AI_BASE_URL", "https://api.deepseek.com")
        self.model_name = os.getenv("AI_MODEL_NAME", "deepseek-chat")
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.error(f"[!] AI連接失敗: {str(e)}")

    def generate_insight(self, profile_path: str, user_query: str) -> dict:

        if not self.client: return {"answer": "AI 未配置", "usage": {}}

        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)

            system_prompt = f"""
            You are a Senior GIS Expert. Analyze the metadata JSON and answer questions.
            Metadata: {json.dumps(profile_data)}
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            
            # Token 使用統計
            usage = response.usage
            return {
                "answer": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                }
            }
        except Exception as e:
            return {"answer": f"Error: {str(e)}", "usage": {}}