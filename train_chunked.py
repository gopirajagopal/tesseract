"""train_chunked.py — Run pre-training in 10k-step chunks for limited hardware.

Finds the latest checkpoint in checkpoints/general/, then runs train.py
for exactly CHUNK_STEPS more steps. Run this script once per chunk —
each invocation does one 10k-step block and stops cleanly.

Usage (run 10 times, or use the loop at the bottom):
    # Single chunk (run manually after each finishes):
    python train_chunked.py

    # Or let it loop all chunks automatically:
    python train_chunked.py --all
"""
import argparse
import subprocess
import sys
from pathlib import Path

CHUNK_STEPS   = 10_000          # steps per chunk
TOTAL_TARGET  = 60_000          # final absolute step goal (10k start + 50k new)
CONFIG        = "configs/general.yaml"
CKPT_DIR      = Path("checkpoints/general")
FIRST_RESUME  = "checkpoints/small/step_00010000"   # base pre-training checkpoint
TOKENIZER     = "tokenizer"


def latest_checkpoint() -> tuple[str, int]:
    """Return (path_str, step) of the latest checkpoint in CKPT_DIR."""
    best_step = -1
    best_path = None
    for p in CKPT_DIR.glob("step_*"):
        try:
            step = int(p.name.split("_")[1])
            if step > best_step:
                best_step = step
                best_path = str(p)
        except (IndexError, ValueError):
            pass
    return best_path, best_step


def run_chunk(resume_path: str, target_step: int) -> int:
    cmd = [
        sys.executable, "train.py",
        "--config",    CONFIG,
        "--resume",    resume_path,
        "--tokenizer", TOKENIZER,
        "--max-steps", str(target_step),
    ]
    print(f"\n{'='*60}")
    print(f"  Chunk: {resume_path} → step {target_step:,}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    return subprocess.run(cmd, check=False).returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="Loop through all remaining chunks automatically")
    args = parser.parse_args()

    if args.all:
        # Run all remaining chunks sequentially
        while True:
            resume, current_step = latest_checkpoint()
            if resume is None:
                resume = FIRST_RESUME
                current_step = 10_000   # the step stored in that checkpoint

            if current_step >= TOTAL_TARGET:
                print(f"\nAll done — reached step {current_step:,}. Pre-training complete!")
                break

            target = min(current_step + CHUNK_STEPS, TOTAL_TARGET)
            rc = run_chunk(resume, target)
            if rc != 0:
                print(f"\nChunk failed (exit {rc}). Fix the error and re-run.")
                sys.exit(rc)

            # Verify the new checkpoint was written
            new_resume, new_step = latest_checkpoint()
            print(f"\nChunk complete. Latest checkpoint: step {new_step:,}")

    else:
        # Single chunk
        resume, current_step = latest_checkpoint()
        if resume is None:
            resume = FIRST_RESUME
            current_step = 10_000

        if current_step >= TOTAL_TARGET:
            print(f"Already at step {current_step:,} — nothing to do.")
            return

        target = min(current_step + CHUNK_STEPS, TOTAL_TARGET)
        rc = run_chunk(resume, target)
        if rc != 0:
            print(f"\nChunk failed (exit {rc}).")
            sys.exit(rc)

        _, new_step = latest_checkpoint()
        remaining = max(0, TOTAL_TARGET - new_step)
        chunks_left = (remaining + CHUNK_STEPS - 1) // CHUNK_STEPS
        print(f"\n  Done. Step: {new_step:,} / {TOTAL_TARGET:,}")
        print(f"  Chunks remaining: {chunks_left}")
        if chunks_left > 0:
            print(f"  Run again: python train_chunked.py")
        else:
            print("  Pre-training complete! Next: run SFT.")
            print("  python finetune.py --stage sft --config configs/general.yaml \\")
            print(f"      --base-checkpoint checkpoints/general/step_{new_step:08d} \\")
            print("      --data data/general_sft_train.jsonl --tokenizer tokenizer \\")
            print("      --checkpoint-dir checkpoints/general_sft")


if __name__ == "__main__":
    main()
