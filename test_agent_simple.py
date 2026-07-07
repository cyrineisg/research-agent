from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from agent.tools import search_arxiv, read_paper_pdf
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # ce modèle gère mieux le tool calling
    api_key=os.getenv("GROQ_API_KEY")
)

tools = [search_arxiv, read_paper_pdf]
llm_with_tools = llm.bind_tools(tools)

message = HumanMessage(
    content="Search for the top papers on LLM agents"
)

response = llm_with_tools.invoke([message])

print("=== RÉPONSE DU LLM ===")
print(response.content)
print("\n=== OUTILS APPELÉS ===")
print(response.tool_calls)