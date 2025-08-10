import os
import io
import re
import json
import base64
import traceback
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="Data Analyst Agent API")

# --- System Prompt for the Planner LLM ---
PLANNER_SYSTEM_PROMPT = """
You are a world-class data analyst AI agent. Your task is to create a Python execution plan in JSON format to answer the user's questions about data.

**Environment and Rules:**

- The data is pre-loaded into a pandas DataFrame named `df`.
- Your first step MUST clean and prepare the data, returning the cleaned DataFrame as the last expression.
- Subsequent steps MUST reference this cleaned DataFrame as `result_0`.
- Each step returns a value stored as `result_i` (step 0 = `result_0`, step 1 = `result_1`, etc.).
- For plotting steps, generate the plot using matplotlib/seaborn WITHOUT saving files or base64 encoding.
- The system automatically captures plot images and encodes them.
- Your JSON output format is:
  {
    "plan": [
      {"tool": "python", "code": "..."},
      {"tool": "python", "code": "..."},
      ...
    ]
  }

- The last step should return a JSON-compatible dictionary or list with the final answers referencing previous `result_i` values.

**Code guidelines:**

- Ensure all blocks (`if`, `try`, etc.) have properly indented bodies.
- Use `result_0` to refer to the cleaned DataFrame in later steps.
- Return values explicitly as the last line in each code block.

---

**Important Data Cleaning Instructions:**

- When working with scraped tabular data, numeric columns may contain non-numeric characters such as currency symbols (e.g., '$'), commas, spaces, and citation/reference markers (e.g., '[1]').
- Before performing any numerical calculations, you MUST clean these columns by:
  - Removing all non-numeric characters (including '$', ',', spaces, and any bracketed citations like '[1]', '[a]', etc.).
  - Converting the cleaned strings to appropriate numeric types using `pd.to_numeric(..., errors='coerce')`.
- For date or year columns, extract only the 4-digit year and convert to integers.
- After cleaning, drop rows where essential numeric columns have missing values (NaNs) caused by the conversion.
- Ensure all relevant columns used in calculations are clean, numeric, and free of string artifacts.

**This cleaning step should be done immediately after loading the data into the DataFrame `df` and before any analysis or plotting.**


Example plan for the request:

- Step 0: Clean and prepare the DataFrame `df` and return it.
- Step 1: Compute count of movies grossing over $2B before 2000.
- Step 2: Find earliest movie over $1.5B.
- Step 3: Calculate correlation between Rank and Peak.
- Step 4: Draw scatterplot with regression line.
- Step 5: Return a JSON array with all answers and the plot image URI.

Analyze the user's request and data context, then generate the JSON plan.
"""

def safe_loads(json_like: str):
    """Convert Gemini's JSON-like output to valid JSON and parse it."""
    # Remove leading/trailing ``` blocks
    json_like = json_like.strip().strip("```").strip()
    # Escape unescaped newlines inside string values
    def escape_newlines(match):
        return match.group(0).replace("\n", "\\n")
    json_like = re.sub(r'"code":\s*"([^"]*?)"', lambda m: f'"code": "{m.group(1).replace("\n", "\\n")}"', json_like, flags=re.DOTALL)
    return json.loads(json_like)

def load_and_clean_url_table(url: str) -> pd.DataFrame:
    """Download HTML tables from URL and return cleaned first table as a DataFrame."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        raise ValueError(f"No tables found at URL: {url}")
    
    df_raw = tables[0]

    # Example cleaning - you can customize this per your data
    # Remove citation references like '[a]', '[1]' from columns
    df_raw.columns = [re.sub(r'\s*\[[^\]]*\]', '', col).strip() for col in df_raw.columns]

    # Further cleaning depending on your table structure goes here...
    
    return df_raw

# --- Code Execution Sandbox ---
class CodeExecutor:
# """A stateful and safe environment to execute LLM-generated Python code."""
    def __init__(self, df: pd.DataFrame):
        self.exec_globals = {
            "pd": pd, "np": np, "plt": plt,
            "sns": sns, "stats": stats, "df": df
        }
        self.execution_scope = {}

    

    def execute_step(self, code: str):
        try:
            # --- Clean LLM quirks ---
            code = "\n".join(line.lstrip() for line in code.splitlines())  # remove stray indents
            code = code.replace("plt.show()", "")  # remove plt.show()
            
            if any(c in code for c in ["plt.", "sns."]):
                result = self._execute_plot_code(code)
            else:
                result = self._execute_data_code(code)
            
            return self._convert_to_json_serializable(result)
        except Exception as e:
            print(f"Error executing code:\n{code}\nError: {e}")
            traceback.print_exc()
            return f"Execution failed with error: {e}"

    def _execute_data_code(self, code: str):
        """Executes Python code that produces a value."""
        # Try evaluating the entire block as an expression first (for dicts/lists/multi-line returns)
        try:
            compiled = compile(code, '<string>', 'eval')
            return eval(compiled, self.exec_globals, self.execution_scope)
        except SyntaxError:
            pass  # Not a pure expression, fall back to exec + eval of last expression

        # If not a pure expression, split into statements + final expression
        exec_lines = [line for line in code.strip().split('\n') if line.strip()]
        if not exec_lines:
            return None

        if len(exec_lines) > 1:
            exec("\n".join(exec_lines[:-1]), self.exec_globals, self.execution_scope)
            return eval(exec_lines[-1], self.exec_globals, self.execution_scope)
        else:
            exec(code, self.exec_globals, self.execution_scope)
            return None

    def _execute_plot_code(self, code: str):
        """Executes plotting code and returns the base64 data URI."""
        plt.close('all')
        exec(code, self.exec_globals, self.execution_scope)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        data_uri = f"data:image/png;base64,{base64_str}"
        if len(data_uri) > 100_000:
            buf = io.BytesIO()
            plt.savefig(buf, format='webp', bbox_inches='tight', dpi=70, quality=75)
            buf.seek(0)
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            data_uri = f"data:image/webp;base64,{base64_str}"
        plt.close('all')
        return data_uri

    def _convert_to_json_serializable(self, obj):
        """Converts numpy/pandas objects to JSON-serializable Python types."""
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, dict): return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [self._convert_to_json_serializable(i) for i in obj]
        return obj

    # --- API Endpoints ---
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/")
    async def analyze_data(
        questions: UploadFile = File(...),
        # FIX: Use `Optional` and `default=None` to correctly handle missing files
        data: Optional[UploadFile] = File(default=None)
    ):
        try:
            question_text = (await questions.read()).decode("utf-8")
            df, data_context = None, ""

            # FIX: The condition `if data and data.filename` correctly handles the optional file
            if data and data.filename:
                csv_content = (await data.read()).decode("utf-8")
                df = pd.read_csv(io.StringIO(csv_content))
                data_context = f"User uploaded a CSV. Columns: {list(df.columns)}"
            else:
                url_match = re.search(r'https?://[^\s<>"\'()]+', question_text)
                if url_match:
                    url = url_match.group(0)
                    try:
                        df = load_and_clean_url_table(url)
                        data_context = f"Data scraped from {url}. Columns: {list(df.columns)}"
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"Failed to process URL: {url}. Error: {str(e)}")
                else:
                    raise HTTPException(status_code=400, detail="No data file provided and no URL in question.")

            if df is None:
                raise HTTPException(status_code=500, detail="DataFrame could not be loaded.")

            full_prompt = f"{data_context}\n\nUser request:\n```\n{question_text}\n```\n\nGenerate the JSON plan."
            
            model = genai.GenerativeModel('gemini-2.5-pro', system_instruction=PLANNER_SYSTEM_PROMPT)
            response = model.generate_content(full_prompt)

            cleaned_response_text = response.text.strip().lstrip("```json").rstrip("```").strip()
            # plan = json.loads(cleaned_response_text)
            try:
                plan = safe_loads(cleaned_response_text)
            except json.JSONDecodeError:
                print("Failed to decode LLM response as JSON:")
                print(cleaned_response_text)
                raise HTTPException(status_code=500, detail="LLM did not return a valid JSON plan.")
            executor = CodeExecutor(df)
            results = []
            for i, step in enumerate(plan.get("plan", [])):
                if step.get("tool") == "python":
                    result = executor.execute_step(step["code"])
                    results.append(result)
                    # FIX: Store result in the execution scope for later steps to use
                    executor.execution_scope[f'result_{i}'] = result
                
            final_response = results[-1] if results else []
            if "JSON array" in question_text.lower():
                final_response = results

            return JSONResponse(content=final_response)

        except json.JSONDecodeError:
            print("Failed to decode LLM response as JSON:")
            print(cleaned_response_text)
            raise HTTPException(status_code=500, detail="LLM did not return a valid JSON plan.")
        except Exception as e:
            print("An unhandled exception occurred:")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

    if __name__ == "__main__":
        import uvicorn
        # Use `reload=True` for easier development
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)