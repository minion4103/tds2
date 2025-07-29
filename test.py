import asyncio
from app.llm import ask_llm

async def test():
    try:
        response = await ask_llm("Hello, which is the best free ai agent model")
        print("Response:", response)
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
