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
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [search_arxiv, read_paper_pdf]
llm_with_tools = llm.bind_tools(tools)

# ============================================
# 3. PROMPT SYSTÈME
# ============================================
SYSTEM_PROMPT = """You are a research assistant that helps users find and summarize academic papers.
When given a topic or questipn: 
1. Use search_arxiv to find relevant papers
2. Use read_paper_pdf to read the most relevant ones
3. Provide a clear, structured summary with key findings

Always cite paper titles and authors in your response"""

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