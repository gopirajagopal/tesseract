"""run_after_pretrain.py — Auto-run SFT then restart server after pre-training.

Waits for the latest checkpoint in checkpoints/general/ to reach
the target step, then immediately kicks off SFT fine-tuning on
data/general_sft_train.jsonl and restarts the web server.

Usage (run in a separate terminal while train.py is running):
    python run_after_pretrain.py
"""
import subprocess
import sys
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_DIR       = Path("checkpoints/general")
TARGET_STEP    = 100_000      # wait until this step exists
POLL_INTERVAL  = 60           # seconds between checks

SFT_DATA       = "data/general_sft_train.jsonl"
SFT_CKPT_DIR   = "checkpoints/general_sft"
CONFIG         = "configs/general.yaml"
TOKENIZER      = "tokenizer"
SERVER_PORT    = 8000


def latest_step(ckpt_dir: Path) -> int:
    """Return the highest step number found in the checkpoint directory."""
    steps = []
    for p in ckpt_dir.glob("step_*"):
        try:
            steps.append(int(p.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(steps) if steps else 0


def find_checkpoint(ckpt_dir: Path, step: int) -> Path | None:
    for p in ckpt_dir.glob(f"step_{step:08d}"):
        return p
    return None


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, check=False)
    return result.returncode


# ── Phase 1: wait for pre-training to finish ──────────────────────────────────
print(f"Watching {CKPT_DIR} for step {TARGET_STEP:,} …")
print("(Press Ctrl+C to abort)\n")

while True:
    step = latest_step(CKPT_DIR)
    print(f"\r  Latest checkpoint: step {step:,} / {TARGET_STEP:,}", end="", flush=True)

    if step >= TARGET_STEP:
        print(f"\n  Pre-training complete at step {step:,}!")
        break

    time.sleep(POLL_INTERVAL)

ckpt = find_checkpoint(CKPT_DIR, TARGET_STEP) or CKPT_DIR / f"step_{TARGET_STEP:08d}"

# ── Phase 2: SFT fine-tuning ──────────────────────────────────────────────────
rc = run([
    sys.executable, "finetune.py",
    "--stage",            "sft",
    "--config",           CONFIG,
    "--base-checkpoint",  str(ckpt),
    "--data",             SFT_DATA,
    "--tokenizer",        TOKENIZER,
    "--checkpoint-dir",   SFT_CKPT_DIR,
], "Phase 2: SFT fine-tuning on Dolly-15k + Alpaca")

if rc != 0:
    print(f"SFT failed (exit code {rc}). Fix errors then restart the server manually.")
    sys.exit(rc)

# ── Phase 3: find best SFT checkpoint ─────────────────────────────────────────
sft_dir = Path(SFT_CKPT_DIR)
sft_step = latest_step(sft_dir)
sft_ckpt = sft_dir / f"step_{sft_step:08d}"
print(f"\nBest SFT checkpoint: {sft_ckpt}")

# ── Phase 4: restart web server ───────────────────────────────────────────────
print("\nRestarting web server with general_sft checkpoint …")
subprocess.Popen([
    sys.executable, "server.py",
    "--checkpoint", str(sft_ckpt),
    "--config",     CONFIG,
    "--tokenizer",  TOKENIZER,
    "--port",       str(SERVER_PORT),
])

print(f"\nServer started on http://localhost:{SERVER_PORT}")
print("Done! The model now has general English + instruction-following ability.")
