import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
import asyncio
import requests
from bs4 import BeautifulSoup
import duckdb
import seaborn as sns

def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_np_types(i) for i in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    else:
        return obj

# Define allowed modules for safe execution
ALLOWED_MODULES = {
    'plt': plt,
    'io': io,
    'base64': base64,
    'np': np,
    'pd': pd,
    'requests': requests,
    'BeautifulSoup': BeautifulSoup,
    'duckdb': duckdb,
    'sns': sns,
}

# Restricted built-ins for security
RESTRICTED_BUILTINS = {
    'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
    'bin': bin, 'bool': bool, 'bytearray': bytearray, 'bytes': bytes,
    'chr': chr, 'complex': complex, 'dict': dict, 'dir': dir,
    'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
    'float': float, 'format': format, 'frozenset': frozenset,
    'hash': hash, 'hex': hex, 'int': int, 'iter': iter,
    'len': len, 'list': list, 'map': map, 'max': max,
    'min': min, 'next': next, 'oct': oct, 'ord': ord,
    'pow': pow, 'print': print, 'range': range, 'repr': repr,
    'reversed': reversed, 'round': round, 'set': set,
    'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum,
    'tuple': tuple, 'zip': zip, 'Exception': Exception,
    'ValueError': ValueError, 'TypeError': TypeError,
    'KeyError': KeyError, 'IndexError': IndexError,
    'AttributeError': AttributeError, 'ZeroDivisionError': ZeroDivisionError,
    'ImportError': ImportError, 'ModuleNotFoundError': ModuleNotFoundError,
}

def safe_execute_sync(code: str) -> dict:
    try:
        local_vars = {}
        env = {**RESTRICTED_BUILTINS, **ALLOWED_MODULES}
        exec(code, env, local_vars)
        result = local_vars.get("result", {"error": "No result returned."})
        
        if hasattr(result, "to_dict"):
            result = result.to_dict(orient='records')
            
        result = convert_np_types(result)
        return result
    except Exception as e:
        return {"error": str(e)}

# ... (keep all the existing code until the safe_execute function)

async def safe_execute(code: str) -> dict:
    try:
        # Use a timeout that's slightly less than Vercel's max duration
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            try:
                result = await asyncio.wait_for(
                    loop.run
                    (
                        lambda: pool.submit(safe_execute_sync, code),
                        timeout=30.0
                    ),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                return {"error": "Execution timed out."}
    
        return result
    except Exception as e:
        return {"error": str(e)}