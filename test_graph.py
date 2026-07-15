from agent.graph import app_graph
from langchain_core.messages import HumanMessage

print("🚀 Lancement...")

for step in app_graph.stream(
    {"messages": [HumanMessage(content="Search for 2 papers on transformers. Do NOT read any PDF, just search.")]},
    stream_mode="updates"
):
    node_name = list(step.keys())[0]
    print(f"⚙️ Étape : [{node_name}]")
    messages = step[node_name].get("messages", [])
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"   🔧 {tc['name']}({tc['args']})")
        elif msg.content:
            print(f"   💬 {msg.content[:300]}")