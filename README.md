# LangGraph ReAct Agent 🤖

A ReAct-pattern AI agent built with LangGraph, LangChain, and Groq.

## 🛠️ Tech Stack
- LangGraph
- LangChain
- Groq + LLaMA 3.3 70B
- DuckDuckGo Search Tool
- MemorySaver (conversation memory)

## 📁 Project Structure
- `agent.py` – main agent logic with LangGraph
- `tools.py` – custom tools (web search, calculator, save_note)
- `config.py` – model and API configuration
- `run.py` – entry point to run the agent

## ⚙️ Setup
1. Clone the repo
2. Create `.env` file with `GROQ_API_KEY=your_key`
3. Install: `pip install -r requirements.txt`
4. Run: `python run.py`
