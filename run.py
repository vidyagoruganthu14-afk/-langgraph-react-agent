from langchain_core.messages import HumanMessage
from agent import build_agent
# Build the agent (compiles the LangGraph StateGraph)
app = build_agent()
# WHY thread_id:
#   This is the "conversation session ID".
#   Same thread_id  → agent remembers the full conversation history.
#   Different thread_id → fresh conversation, no memory of past turns.
#   In production, generate a unique ID per user/session.
config = {"configurable": {"thread_id": "session-001"}}
print("=" * 50)
print("  LangGraph Agent — Ready!")
print("  Type 'quit' or 'exit' to stop.")
print("  Type 'new' to start a fresh conversation.")
print("=" * 50)
print()
# Main conversation loop
session_counter = 1
while True:
    #Get user input
    try:
        user_input = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break
    # Handle special commands
    if user_input.lower() in ("quit", "exit"):
        print("Goodbye!")
        break
    if user_input.lower() == "new":
        # Start a fresh conversation by switching to a new thread_id.
        # WHY: MemorySaver stores history per thread_id. A new ID =
        # a blank slate with no memory of previous messages.
        session_counter += 1
        config = {"configurable": {"thread_id": f"session-{session_counter:03d}"}}
        print(f"[Started new conversation: session-{session_counter:03d}]\n")
        continue
    if not user_input:
        continue
    # Stream the agent response
    # WHY stream instead of invoke:
    #   stream() yields state snapshots after each node runs.
    #   This lets us detect mid-graph pauses (human-in-the-loop)
    #   and also lets us show output progressively.
    # stream_mode="values" gives us the FULL state at each step,
    # not just the delta. Easier to read the last message from.
    print("\nAgent: ", end="", flush=True)
    try:
        final_state = None
        for chunk in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values",
        ):
            final_state = chunk
        # ── Safety check ───────────────────────────────────────────
        if final_state is None:
            print("No response received. Please try again.")
            print()
            continue
        last_msg = final_state["messages"][-1]
        #Human-in-the-loop handling
        # WHY: When interrupt_before=["tools"] is set in agent.py,
        # the graph PAUSES before running the tool node and returns
        # control here. The last message will contain tool_calls
        # showing WHAT the agent wants to do. We ask for approval.
        # If approved  → call app.stream(None, config) to RESUME.
        #   Passing None means "no new input, just continue from
        #   the saved checkpoint."
        # If rejected  → skip the tool, inform the user.
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tool_name = last_msg.tool_calls[0]["name"]
            tool_args = last_msg.tool_calls[0]["args"]
            print(f"\n[Agent wants to use tool: '{tool_name}']")
            print(f"[Arguments: {tool_args}]")
            approval = input("Approve this tool call? (y/n): ").strip().lower()
            if approval == "y":
                print("\nAgent: ", end="", flush=True)
                # Resume the graph from the checkpoint
                for chunk in app.stream(
                    None,
                    config=config,
                    stream_mode="values",
                ):
                    final_state = chunk
                print(final_state["messages"][-1].content)
            else:
                print(
                    "Agent: Understood, I won't use that tool. "
                    "Let me try to answer without it."
                )
        else:
            # ── Normal response (no tool call) ─────────────────────
            print(last_msg.content)
    # Global error handler
    # WHY: Groq sometimes returns APIError for malformed tool calls
    # (e.g. "book a ticket" triggers web_search with bad JSON).
    # Instead of crashing the whole app, we catch it, show a friendly
    # message, and let the user try again. The agent stays alive.
    except Exception as e:
        print(
            f"\n[Error: {e}]\n"
            "Agent: Sorry, something went wrong processing that request. "
            "Please try rephrasing your question."
        )
    print()