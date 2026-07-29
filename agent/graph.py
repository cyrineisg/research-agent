from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Annotated 
from agent.tools import search_arxiv, read_paper_pdf
from dotenv import load_dotenv
import operator
from pathlib import Path
import os
from agent.tools import search_arxiv, read_paper_pdf, check_memory, export_summary


load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ============================================
# 1. STATE — ce que l'agent mémorise
# ============================================
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# ============================================
# 2. LLM + OUTILS
# ============================================
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [check_memory, search_arxiv, read_paper_pdf, export_summary]
llm_with_tools = llm.bind_tools(tools)

# ============================================
# 3. PROMPT SYSTÈME
# ============================================
SYSTEM_PROMPT = """You are a multilingual research assistant specialized in scientific papers.

LANGUAGE RULES:
- If the user writes in Arabic, respond ENTIRELY in Arabic
- If the user writes in French, respond ENTIRELY in French
- If the user writes in English, respond ENTIRELY in English
- Always search arXiv in English regardless of input language

For every request, follow this exact order:
1. ALWAYS call check_memory FIRST to see if you already know about this topic
2. If not in memory, use search_arxiv to find new papers
3. Use read_paper_pdf on the most relevant paper only
4. Provide a clear structured summary in plain text

EXPORT RULES — very important:
- When calling export_summary, pass ONLY plain text with no special characters
- No markdown symbols like ** or ## in the content parameter
- No newline characters \n in the content parameter
- Keep the content parameter under 500 characters
- Use a simple short filename with no spaces"""
# ============================================
# 4. NOEUDS DU GRAPHE
# ============================================
def agent_node(state: AgentState):
    """Le LLM décide quoi faire."""
    messages = state["messages"]

    #Ajoute le prompt système si c'est le premier message
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

#ToolNode exécute automatiquement l'outil si le LLM le demande
tool_node = ToolNode(tools)

# ============================================
# 5. CONDITION : continuer ou terminer ?
# ============================================
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ============================================
# 6. CONSTRUCTION DU GRAPHE
# ============================================
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app_graph = graph.compile()