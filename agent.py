from typing import TypedDict, Annotated
import operator
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from config import GROQ_API_KEY, MODEL_NAME
from tools import ALL_TOOLS
# STEP A: Define State
# WHY: This TypedDict is the ONLY thing shared across all nodes.
# Annotated[list, operator.add] means "append new messages, don't
# replace". If you used just `messages: list`, each node would
# OVERWRITE the full list instead of adding to it.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
# STEP B: Set up the LLM
# WHY Groq: Free, fast (~500 tokens/sec), supports tool calling.
# bind_tools() tells the LLM "these tools exist, you may call them".
# temperature=0 makes output deterministic — best for agents.
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=0,
    streaming=True,
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)
# STEP C: System Prompt
# WHY: Without this the agent has no identity, rules, or boundaries.
# It is prepended to every LLM call so the model always knows its role.
# Clear rules prevent the LLM from misusing tools on ambiguous queries.
SYSTEM_PROMPT = SystemMessage(content="""You are a smart, helpful AI assistant.
You have access to these tools — use them ONLY when clearly needed:
- web_search : ONLY for real-time info, news, current events, prices
- calculator  : ONLY for mathematical calculations
- save_note   : ONLY when the user explicitly asks to save/remember something

IMPORTANT rules:
- Do NOT use web_search for ticket booking, reservations, or transactions.
  You cannot actually book tickets — tell the user you don't have that capability
  and suggest platforms like MakeMyTrip, IRCTC, or RedBus.
- Do NOT use a tool if you can answer from your own knowledge.
- If unsure whether to use a tool, just answer directly without one.
- Be concise and honest. If you cannot do something, say so clearly.
""")
# STEP D: Define Nodes
def llm_node(state: AgentState) -> dict:
    """
    The brain node — calls the LLM with the full conversation history.
    WHY prepend system prompt here (not in state):
      Keeps the state clean. The system prompt never gets stored in
      memory or duplicated across turns.
    WHY fallback:
      Groq/Llama sometimes generates malformed tool-call JSON for
      ambiguous requests. Instead of crashing the whole graph, we
      retry without tools so the user still gets a useful response.
    """
    messages = [SYSTEM_PROMPT] + state["messages"]
    try:
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    except Exception as e:
        # Tool-call generation failed — fall back to plain LLM (no tools).
        # This prevents a single bad query from crashing the entire agent.
        print(f"\n[Debug] Tool call failed, falling back to plain LLM.\nReason: {e}\n")
        fallback_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=0,
            streaming=True,
        )
        response = fallback_llm.invoke(messages)
        return {"messages": [response]}
# WHY ToolNode (prebuilt)?
# Writing your own tool executor is error-prone. LangGraph's ToolNode:
#   - Reads tool_calls from the last message automatically
#   - Finds and executes the matching function
#   - Wraps results in ToolMessage format that the LLM can read
#   - Handles errors inside tools gracefully
# You get all of this for free — one line.
tool_node = ToolNode(ALL_TOOLS)
# STEP E: Conditional Edge — the Router
# WHY: After every LLM call we check: did the LLM request a tool?
#   YES → go to tool_node (execute the tool, come back to LLM)
#   NO  → go to END (the agent is done, return the final answer)
# This single function is what makes the agent loop intelligently
# instead of stopping after one LLM response.
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"   # LLM wants a tool → loop back
    return "end"         # LLM is done → exit graph
# STEP F: Build the Graph
def build_agent():
    graph = StateGraph(AgentState)
    # Register nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)
    # Entry point — every run starts at the LLM node
    graph.set_entry_point("llm")
    # After LLM → run the router
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",   # tool call detected → execute tools
            "end":   END        # no tool call → return final answer
        }
    )
    # After tools → always go back to LLM
    # WHY: The tool result is appended to state["messages"]. The LLM
    # must read that result and decide: call another tool, or answer?
    # This creates the ReAct loop: Reason → Act → Reason → Act → ...
    graph.add_edge("tools", "llm")
    # Memory / Checkpointing
    # WHY MemorySaver:
    #   Saves full state (all messages) after every node execution.
    #   With the same thread_id the agent remembers the whole history.
    #   Swap for SqliteSaver("memory.db") to persist across restarts.
    memory = MemorySaver()
    app = graph.compile(
        checkpointer=memory,
        # Human-in-the-loop
        # WHY interrupt_before=["tools"]:
        #   Pauses the graph BEFORE the tool node runs.
        #   The human can review the tool call and approve or reject it.
        #   Call app.stream(None, config) to resume after approval.
        #   Comment this line out to skip human approval entirely.
        interrupt_before=["tools"],
    )
    return app