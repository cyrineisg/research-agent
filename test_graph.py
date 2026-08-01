from agent.graph import app_graph
from langchain_core.messages import HumanMessage

def run(question):
    print(f"\n🔍 Question : {question}\n" + "="*50)

    for step in app_graph.stream(
        {
            "messages": [HumanMessage(content=question)],
            "papers_found": "",
            "papers_content": "",
            "final_summary": ""
        },
        stream_mode="updates"
    ):
        node_name = list(step.keys())[0]
        print(f"\n⚙️ [{node_name}]")
        messages = step[node_name].get("messages", [])
        for msg in messages or []:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"   🔧 {tc['name']}({tc['args']})")
            elif hasattr(msg, "name") and msg.name:
                print(f"   📄 {msg.name}: {msg.content[:150]}...")
            elif msg.content:
                print(f"   💬 {msg.content[:200]}")

run("Find papers on LoRA fine-tuning")