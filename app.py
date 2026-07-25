import gradio as gr
from agent.graph import app_graph
from langchain_core.messages import HumanMessage

def run_agent(question, history):
    if not question.strip():
        yield "❌ Pose-moi une question sur des papers !"
        return

    print(f"\n🔍 Question reçue : {question}")
    output = ""

    for step in app_graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates"
    ):
        node_name = list(step.keys())[0]
        print(f"⚙️ Étape : [{node_name}]")
        node_data = step[node_name]

        if not node_data:
            continue

        messages = node_data.get("messages", [])

        if not messages:
            continue

        for msg in messages:
            if msg is None:
                continue

            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"   🔧 {tc['name']}({tc['args']})")
                    output += f"🔧 `{tc['name']}` → `{tc['args']}`\n\n"
                yield output

            elif hasattr(msg, "name") and msg.name and msg.content:
                print(f"   📄 Résultat {msg.name} : {msg.content[:100]}...")
                output += f"📄 **{msg.name}** :\n{msg.content[:300]}...\n\n"
                yield output

            elif hasattr(msg, "content") and msg.content:
                tool_calls = getattr(msg, "tool_calls", None)
                if not tool_calls:
                    print(f"   ✅ Réponse finale : {msg.content[:100]}...")
                    output += f"---\n\n✅ **Réponse finale** :\n\n{msg.content}"
                    yield output    

    print("✅ Agent terminé")

with gr.Blocks(title="Research Paper Assistant") as demo:
    gr.Markdown("""
    # 🤖 Research Paper Assistant
    Agent IA qui cherche, lit et résume des papers scientifiques sur arXiv.
    
    **Exemples :**
    - *Find and summarize the top 3 papers on RAG*
    - *What are the latest advances in LLM agents?*
    - *Find papers on fine-tuning LLMs efficiently*
    """)

    gr.ChatInterface(
        fn=run_agent,
        examples=[
            "Find and summarize the top 3 papers on RAG",
            "What are the latest advances in LLM agents?",
            "Find papers on fine-tuning LLMs efficiently"
        ],
    )

demo.launch()