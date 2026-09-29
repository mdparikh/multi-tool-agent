# Gradio UI
import gradio as gr
from agent_app import run_agent_pipeline
import spaces

@spaces.GPU
def gradio_chat_interface(prompt, provider):
    res = run_agent_pipeline(user_prompt=prompt, provider=provider)
    return f"**Status:** {res['status']}\n\n**Response:**\n{res['response']}"

interface = gr.Interface(
    fn=gradio_chat_interface,
    inputs=[
        gr.Textbox(lines=3, placeholder="Ask a question (e.g., 'What is our remote work policy?' or 'SQL query: select * from employees')..."),
        gr.Dropdown(choices=["gemini", "groq", "huggingface"], value="gemini", label="LLM Provider Engine")
    ],
    outputs=gr.Markdown(label="Agent Output"),
    title="Enterprise Guardrailed Multi-Tool AI Agent",
    description="Pipeline Flow: User ➔ Input Guardrails ➔ Agent (SQL, Vector RAG, Search) ➔ Output Guardrails ➔ Langfuse Tracing"
)

interface.launch(ssr_mode=False)