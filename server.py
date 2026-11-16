"""server.py — FastAPI web server for interactive LLM chat.

Serves a streaming chat API (Server-Sent Events) and a static frontend.

Usage:
    python server.py --checkpoint checkpoints/dpo/step_00002257 \
                     --config configs/small.yaml \
                     --tokenizer tokenizer \
                     --port 8000
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import torch
import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from data.tokenizer import BPETokenizer, SentencePieceTokenizer
from inference.generate import Generator
from model.architecture import Transformer
from model.config import ModelConfig
from training.checkpointing import Checkpointer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LLM Chat Web Server")
parser.add_argument("--checkpoint", default="checkpoints/dpo/step_00002257")
parser.add_argument("--config",     default="configs/small.yaml")
parser.add_argument("--tokenizer",  default="tokenizer")
parser.add_argument("--host",       default="0.0.0.0")
parser.add_argument("--port",       type=int, default=8000)
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model loading (at startup)
# ---------------------------------------------------------------------------
print("Loading model…")
with open(args.config, "r") as f:
    raw = yaml.safe_load(f)
model_cfg = ModelConfig(**raw.get("model", {}))

tok_path = Path(args.tokenizer)
if (tok_path / "type.txt").exists():
    tokenizer = SentencePieceTokenizer.load(tok_path)
else:
    tokenizer = BPETokenizer.load(tok_path)
model_cfg.vocab_size = len(tokenizer)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(model_cfg).to(device)

ckpt_path = Path(args.checkpoint)
checkpointer = Checkpointer(str(ckpt_path.parent))
checkpointer.load(model, optimizer=None, scheduler=None, checkpoint_path=str(ckpt_path))
model.to(device)

generator = Generator(model, tokenizer, device=device)
print(f"Model ready on {device}  ({sum(p.numel() for p in model.parameters()):,} params)")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static folder if it exists
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = 0.4   # lower = more focused for small models
    top_p: Optional[float] = 0.85
    top_k: Optional[int] = 40
    repetition_penalty: Optional[float] = 1.3  # higher = less repetition/looping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_prompt(messages: List[Message]) -> str:
    # Format must exactly match InstructionDataset._format_conversation:
    #   full_text = "\n".join([f"{role_token} {content}" for each turn])
    # Encoded as ONE string → identical BPE tokenization at training & inference.
    parts = []
    for msg in messages:
        if msg.role == "user":
            parts.append(f"Human: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


async def token_stream(req: ChatRequest) -> AsyncGenerator[str, None]:
    prompt = build_prompt(req.messages)
    max_tok = req.max_new_tokens or args.max_new_tokens
    loop = asyncio.get_event_loop()

    # Run blocking generator in a thread pool to keep the event loop free
    gen = generator.stream(
        prompt,
        max_new_tokens=max_tok,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        stop_tokens=[tokenizer.eos_token_id],
    )

    for chunk in gen:
        payload = json.dumps({"token": chunk})
        yield f"data: {payload}\n\n"
        await asyncio.sleep(0)   # yield control to event loop

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = Path("static/index.html")
    if html_file.exists():
        return HTMLResponse(
            html_file.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return HTMLResponse("<h1>static/index.html not found</h1>", status_code=404)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        token_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "device": device, "vocab_size": len(tokenizer)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
