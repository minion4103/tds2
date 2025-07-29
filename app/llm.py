import httpx
import os
import re
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

async def ask_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "DataAnalystAgent",
    }

    body = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": [{"role": "user", "content": prompt}],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)

        if res.status_code != 200:
            raise Exception(f"OpenRouter error {res.status_code}: {res.text}")

        data = res.json()

        if "choices" not in data:
            raise Exception(f"'choices' not in response: {data}")

        return data["choices"][0]["message"]["content"]

def extract_code(text: str) -> str:
    pattern = r"```(?:python)?\n([\s\S]+?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()