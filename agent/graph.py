from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated 
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
    papers_found: str        # résultat du chercheur
    papers_content: str      # résultat du lecteur
    final_summary: str       # résultat du synthétiseur

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
# AGENT 1 — CHERCHEUR
# ============================================
search_tools = [check_memory, search_arxiv]
search_llm = llm.bind_tools(search_tools)
search_tools_node = ToolNode(search_tools)

SEARCHER_PROMPT = """You are a research search specialist.
Your ONLY job is to find relevant papers.
1. Call check_memory first
2. If memory has results, stop and return them — do NOT call search_arxiv
3. If memory is empty, call search_arxiv ONCE only
4. Return the papers found — nothing else.
Always search in English."""

def searcher_node(state: AgentState):
    messages = [SystemMessage(content=SEARCHER_PROMPT)] + state["messages"]
    response = search_llm.invoke(messages)
    return {"messages": [response]}

def searcher_should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "search_tools"
    
    # Compte combien de fois search_arxiv a été appelé
    arxiv_attempts = sum(
        1 for msg in state["messages"]
        if hasattr(msg, "name") and msg.name == "search_arxiv"
    )
    
    # Si arXiv a échoué 1 fois → passe directement au reader
    if arxiv_attempts >= 1:
        return "reader"
    
    return "reader"


# ============================================
# AGENT 2 — LECTEUR
# ============================================
read_tools = [read_paper_pdf]
read_llm = llm.bind_tools(read_tools)
read_tools_node = ToolNode(read_tools)

READER_PROMPT = """You are a research paper reader specialist.
Your ONLY job is to read ONE paper — no more.
1. Pick the single most relevant paper from the list
2. Call read_paper_pdf ONCE with its URL
3. STOP immediately after — do not call any other tool
Do NOT call read_paper_pdf more than once."""

def reader_node(state: AgentState):
    # Récupère les résultats du chercheur
    last_content = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            last_content = msg.content
            break

    messages = [
        SystemMessage(content=READER_PROMPT),
        HumanMessage(content=f"Papers found:\n{last_content}\n\nNow read the most relevant one.")
    ]
    response = read_llm.invoke(messages)
    return {"messages": [response]}

def reader_should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        # Compte combien de fois read_paper_pdf a été appelé
        read_attempts = sum(
            1 for msg in state["messages"]
            if hasattr(msg, "name") and msg.name == "read_paper_pdf"
        )
        # Maximum 1 lecture → passe au synthesizer
        if read_attempts >= 1:
            return "synthesizer"
        return "read_tools"
    return "synthesizer"



# ============================================
# AGENT 3 — SYNTHÉTISEUR
# ============================================
synth_tools = [export_summary]
synth_llm = llm.bind_tools(synth_tools)

SYNTHESIZER_PROMPT = """You are a research synthesis specialist.
Your job is to produce a clear, structured summary from the papers read.

Structure your summary as:
## Papers Found
[list of papers]

## Key Contributions
[main findings]

## Summary
[2-3 paragraph synthesis]

Respond in the SAME language as the original user question.
If the user asked to export, call export_summary with plain text only (no markdown symbols)."""

def synthesizer_node(state: AgentState):
    # Collecte tout le contexte
    context = "\n".join([
        msg.content for msg in state["messages"]
        if hasattr(msg, "content") and msg.content
        and not (hasattr(msg, "tool_calls") and msg.tool_calls)
    ])

    original_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            original_question = msg.content
            break

    messages = [
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=f"Original question: {original_question}\n\nContext:\n{context[:3000]}\n\nNow synthesize.")
    ]
    response = synth_llm.invoke(messages)
    return {"messages": [response]}

def synthesizer_should_continue(state: AgentState):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "synth_tools"
    return END


# ============================================
# CONSTRUCTION DU GRAPHE
# ============================================
graph = StateGraph(AgentState)

# Noeuds
graph.add_node("searcher", searcher_node)
graph.add_node("search_tools", search_tools_node)
graph.add_node("reader", reader_node)
graph.add_node("read_tools", read_tools_node)
graph.add_node("synthesizer", synthesizer_node)
graph.add_node("synth_tools", ToolNode(synth_tools))

# Flux
graph.set_entry_point("searcher")
graph.add_conditional_edges("searcher", searcher_should_continue)
graph.add_edge("search_tools", "searcher")
graph.add_conditional_edges("reader", reader_should_continue)
graph.add_edge("read_tools", "reader")
graph.add_conditional_edges("synthesizer", synthesizer_should_continue)
graph.add_edge("synth_tools", "synthesizer")

app_graph = graph.compile()