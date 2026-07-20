"""
Core logic for AskMe - conversational CSV analysis.

Kept simple for the first working version:
- extract_schema: summarize a dataframe so the LLM knows what it's working with
- generate_code: ask the LLM for a single pandas expression
- is_safe_code: a basic guardrail check before we execute anything
- run_query: tie it all together
"""

import os
import io
import ast
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)

# --- 1. Schema extraction -----------------------------------------------

def extract_schema(df: pd.DataFrame) -> str:
    """Produce a short text summary of the dataframe for the LLM prompt."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    sample = df.head(3).to_string()
    return f"Columns and dtypes:\n{info_str}\n\nFirst 3 rows:\n{sample}"


# --- 2. Prompt + LLM call -----------------------------------------------

SYSTEM_PROMPT = """You are a data analysis assistant. You are given a pandas \
DataFrame called `df` and a user question about it.

Respond with ONLY a single pandas expression (using `df`) that answers the \
question when evaluated. No explanation, no markdown, no imports, no \
assignment, no print statements - just the raw expression.

Example:
Question: What is the average age?
Answer: df['age'].mean()
"""


def generate_code(question: str, schema: str) -> str:
    user_content = f"Data summary:\n{schema}\n\nQuestion: {question}"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    code = response.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()
    return code


# --- 3. Guardrails -----------------------------------------------------

# Only allow a single expression - no statements, imports, or function defs.
# This blocks things like `import os`, `df.to_csv(...)`, exec/eval calls, etc.
DISALLOWED_NAMES = {
    "__import__", "exec", "eval", "open", "os", "sys", "subprocess",
    "globals", "locals", "compile", "input", "breakpoint",
}


def is_safe_code(code: str) -> tuple[bool, str]:
    """Very basic guardrail: parse as a single expression, block dangerous names."""
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False, "Generated code is not a valid single expression."

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DISALLOWED_NAMES:
            return False, f"Generated code references disallowed name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr in {
            "to_csv", "to_excel", "to_sql", "to_pickle", "eval", "query",
        }:
            return False, f"Generated code calls disallowed method: {node.attr}"

    return True, ""


# --- 4. Framing the answer in natural language -------------------------

EXPLAIN_PROMPT = """You are a data analysis assistant. A user asked a \
question about their data, and pandas code was run to get a result. \
Answer the user's question in one short, plain-English sentence using \
that result. Do not mention code, pandas, or dataframes - just state \
the answer naturally, as if you computed it yourself."""


def explain_result(question: str, result) -> str:
    user_content = f"Question: {question}\nResult: {result}"

    response = llm.invoke([
        SystemMessage(content=EXPLAIN_PROMPT),
        HumanMessage(content=user_content),
    ])
    return response.content.strip()


# --- 5. Execution -----------------------------------------------------

def run_query(df: pd.DataFrame, question: str):
    """Full pipeline: schema -> LLM -> guardrail -> execute -> explain.
    Returns (answer, result, code, error)."""
    schema = extract_schema(df)
    code = generate_code(question, schema)

    safe, reason = is_safe_code(code)
    if not safe:
        return None, None, code, f"Blocked by guardrail: {reason}"

    try:
        result = eval(code, {"df": df, "__builtins__": {}}, {})
    except Exception as e:
        return None, None, code, f"Error running generated code: {e}"

    answer = explain_result(question, result)
    return answer, result, code, None