import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import uvicorn

# NEW: DB imports
from sqlalchemy import create_engine, text

# Expect something like: postgresql://user:pass@host:5432/dbname
DATA_BASE_URL = os.getenv("DATA_BASE_URL")

engine = None
if DATA_BASE_URL:
    engine = create_engine(DATA_BASE_URL, future=True)

    # Create simple tests table if it doesn't exist
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS tests (
                    id SERIAL PRIMARY KEY,
                    ts TIMESTAMP NOT NULL DEFAULT NOW()
                )
                """
            )
        )

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot_small-90M")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(request: ChatRequest):
    inputs = tokenizer(request.user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=100)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return {"user": request.user_input, "bot": reply}

# Simple insert endpoint
@app.post("/insert_test")
def insert_test():
    if engine is None:
        return {"error": "DATA_BASE_URL is not configured"}

    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO tests (ts) VALUES (NOW()) RETURNING id, ts")
        )
        row = result.fetchone()
    return {"id": row.id, "ts": row.ts.isoformat()}

def gradio_chat(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=100)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

with gr.Blocks(title="BlenderBot Chat") as gradio_app:
    gr.Markdown(
        """
        Welcome to your locally hosted AI chatbot powered by FastAPI 
  
        Type your message 
        """,
        elem_id="header"
    )
  
    with gr.Row():
        with gr.Column(scale=4):
            user_input = gr.Textbox(
                label="Your Message",
                placeholder="Type something...",
                lines=3,
                elem_id="input-box"
            )
        with gr.Column(scale=1):
            send_btn = gr.Button("Send", elem_id="send-button")
  
    bot_output = gr.Textbox(
        label="Bot Response",
        lines=5,
        interactive=False,
        elem_id="output-box"
    )
  
    send_btn.click(fn=gradio_chat, inputs=user_input, outputs=bot_output)

gradio_app.css = """
#header {
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 20px;
    color: #333;
}
  
#input-box textarea {
    font-size: 1rem;
    border: 2px solid #4CAF50;
    border-radius: 8px;
    padding: 10px;
    background-color: black;
}
  
#send-button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 60px;
    width: 100%;
    margin-top: 10px;
}
  
#output-box textarea {
    font-size: 1rem;
    background-color: black;
    border-radius: 8px;
    padding: 10px;
    border: 1px solid #ccc;
}
"""

app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    # Make sure DATA_BASE_URL is set in your environment before running
    uvicorn.run(app, host="0.0.0.0", port=8000)
