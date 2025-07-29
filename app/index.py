from fastapi import FastAPI, File, UploadFile
from app.llm import ask_llm, extract_code
from app.executor import safe_execute
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.post("/api/")
async def analyze(file: UploadFile = File(...)):
    question = (await file.read()).decode("utf-8")

    prompt = f"""
You are a data scientist AI.

Your task is to write Python code that:
- Loads data (from the web, from a file, or from a database as required by the question)
- Analyzes the data
- Produces the final answer in the exact format requested (e.g., a JSON array, a JSON object, etc.)
- If a plot is requested, create the plot and encode it as a base64 string in the format "data:image/png;base64,<base64-string>", and include it in the result.

Assign your final answer to a variable named `result`.

Important notes:
- For web scraping, use the `requests` and `BeautifulSoup` libraries.
- For large datasets, use efficient methods (like DuckDB for querying parquet files from S3).
- For plots, use matplotlib and encode the plot as base64.

DO NOT include any explanations, comments, or text outside the code block.

Respond ONLY inside a Python code block (triple backticks).

Question:
{question}
    """

    raw_code = await ask_llm(prompt)
    code = extract_code(raw_code)
    print("Extracted code from LLM:")
    print(code)
    output = await safe_execute(code)
    print("Execution output:")
    print(output)
    return output