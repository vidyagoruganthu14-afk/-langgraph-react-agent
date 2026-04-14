from langchain_core.tools import tool
from duckduckgo_search import DDGS
# TOOL 1: Web Search
# WHY: LLMs have a knowledge cutoff date — they don't know what
# happened yesterday. web_search gives the agent real-time data.
# HOW it works:
#   - The @tool decorator registers this as a LangChain tool.
#   - The docstring is what the LLM reads to decide WHEN to use it.
#   - A clear, specific docstring = fewer wrong tool calls.
# WHY DuckDuckGo:
#   - Completely free, no API key needed.
#   - Good enough for most search tasks.
#   - Swap for Tavily (paid) for higher quality results.
@tool
def web_search(query: str) -> str:
    """Search the web for current, real-time information.
    Use this ONLY when the user asks about recent events, latest news,
    current prices, live scores, or anything that needs up-to-date data.
    Do NOT use this for general knowledge questions you already know."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No search results found. Please try a different query."
        output = ""
        for i, r in enumerate(results, 1):
            output += f"Result {i}:\n"
            output += f"  Title   : {r['title']}\n"
            output += f"  Summary : {r['body']}\n"
            output += f"  URL     : {r['href']}\n\n"
        return output.strip()
    except Exception as e:
        return f"Web search failed: {str(e)}. Please try again."
# TOOL 2: Calculator
# WHY: LLMs are unreliable at arithmetic — they hallucinate numbers.
# A real Python eval() is always correct. Always route math here.
# WHY the safety check:
#   eval() can execute arbitrary Python code, which is dangerous.
#   We whitelist only safe characters (digits, operators, brackets)
#   to prevent code injection attacks like eval("__import__('os')...").
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression accurately.
    Use this for ANY arithmetic, percentage, or numerical calculation.
    Input must be a valid math expression like '23 * 47 + 100 / 5'.
    Do NOT include words or units — numbers and operators only."""
    try:
        # Security: only allow safe math characters
        allowed_chars = set("0123456789+-*/().,% ")
        if not all(c in allowed_chars for c in expression):
            return (
                "Error: expression contains invalid characters. "
                "Use only digits and operators like +, -, *, /, (, )."
            )
        result = eval(expression)
        # Round floats to avoid ugly output like 0.30000000000000004
        if isinstance(result, float):
            result = round(result, 10)
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as e:
        return f"Calculation error: {str(e)}"
# TOOL 3: Save Note
# WHY: Demonstrates how an agent can produce real-world side effects —
# writing to files, databases, or external APIs.
# This is a simple example of a "write" action tool.
# In production you would replace the file write with:
#   - A database INSERT
#   - A Notion / Google Docs API call
#   - A CRM update
@tool
def save_note(content: str) -> str:
    """Save a note or important information to a local file.
    Use this ONLY when the user explicitly says 'save', 'remember',
    'note this down', or similar. Do not use it without clear intent."""
    try:
        with open("agent_notes.txt", "a", encoding="utf-8") as f:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}]\n{content}\n{'─' * 40}\n")
        preview = content[:80] + ("..." if len(content) > 80 else "")
        return f"Note saved successfully at {timestamp}.\nPreview: \"{preview}\""
    except Exception as e:
        return f"Failed to save note: {str(e)}"
# Export
# This list is what gets passed to llm.bind_tools() in agent.py
# and to ToolNode(). Add any new tools here to register them.
ALL_TOOLS = [web_search, calculator, save_note]