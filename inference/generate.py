"""Autoregressive text generation with KV-cache and streaming support."""
from typing import Generator, Iterator, List, Optional

import torch
import torch.nn as nn

from inference.sampling import top_k_top_p_sample


class Generator:
    """Wraps the transformer model for text generation.

    Features:
    - KV-cache for efficient incremental decoding
    - Greedy / temperature / top-k / top-p sampling
    - Repetition penalty
    - Streaming token-by-token output
    - Batch generation
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
    ) -> str:
        """Generate text from a prompt string.

        Args:
            prompt:             Input text.
            max_new_tokens:     Maximum tokens to generate.
            temperature:        Sampling temperature (0 = greedy).
            top_k:              Top-K cutoff (0 = disabled).
            top_p:              Nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: >1.0 penalises repeated tokens.
            stop_tokens:        Token IDs that stop generation.

        Returns:
            Generated string (excluding the prompt).
        """
        tokens = list(self._generate_tokens(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        ))
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    @torch.no_grad()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[List[int]] = None,
    ) -> Iterator[str]:
        """Stream text generation token-by-token.

        Yields decoded string fragments as tokens are produced.
        """
        all_new_ids: List[int] = []
        prev_text = ""
        for token_id in self._generate_tokens(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        ):
            all_new_ids.append(token_id)
            # Decode full sequence and emit only the new suffix.
            # This preserves inter-token spaces that BPE/SP markers encode.
            text = self.tokenizer.decode(all_new_ids, skip_special_tokens=True)
            delta = text[len(prev_text):]
            if delta:
                yield delta
                prev_text = text

    def _generate_tokens(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_tokens: Optional[List[int]],
    ) -> Iterator[int]:
        """Core token generation loop with KV-cache."""
        stop_ids = set(stop_tokens or [self.tokenizer.eos_token_id])

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        all_ids = list(input_ids)

        past_key_values = None
        current_input = input_tensor   # full prompt on first step

        for _ in range(max_new_tokens):
            outputs = self.model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.get("past_key_values")
            logits = outputs["logits"][:, -1, :]   # (1, vocab_size)

            # Sample next token
            next_token = top_k_top_p_sample(
                logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                input_ids=torch.tensor([all_ids], device=self.device),
            )
            next_id = next_token.item()

            if next_id in stop_ids:
                break

            all_ids.append(next_id)
            # After first step, only feed the single new token (KV-cache holds the rest)
            current_input = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            yield next_id

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[str]:
        """Generate responses for a batch of prompts (padded batch, no KV-cache reuse)."""
        encoded = self.tokenizer.batch_encode(
            prompts,
            max_length=self.model.config.max_seq_len - max_new_tokens,
            padding=True,
            truncation=True,
            add_bos=True,
            add_eos=False,
        )
        input_ids      = torch.tensor(encoded["input_ids"],      device=self.device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=self.device)

        B, T = input_ids.shape
        finished     = torch.zeros(B, dtype=torch.bool, device=self.device)
        all_new_ids  = torch.full((B, max_new_tokens), self.tokenizer.pad_token_id,
                                   dtype=torch.long, device=self.device)
        past_key_values = None

        for step in range(max_new_tokens):
            if finished.all():
                break

            if past_key_values is None:
                current_input = input_ids
                current_mask  = attention_mask
            else:
                current_input = all_new_ids[:, step - 1 : step]
                current_mask  = None

            outputs = self.model(
                current_input,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.get("past_key_values")
            logits = outputs["logits"][:, -1, :]

            next_tokens = top_k_top_p_sample(
                logits, top_k=top_k, top_p=top_p, temperature=temperature
            )
            finished |= (next_tokens == self.tokenizer.eos_token_id)
            next_tokens = next_tokens.masked_fill(finished, self.tokenizer.pad_token_id)
            all_new_ids[:, step] = next_tokens

        results = []
        for i in range(B):
            ids = all_new_ids[i].tolist()
            # Trim at first EOS / pad
            for j, t in enumerate(ids):
                if t in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id):
                    ids = ids[:j]
                    break
            results.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return results


def chat(
    model: nn.Module,
    tokenizer,
    system_prompt: str = "You are a helpful, harmless, and honest AI assistant.",
    device: Optional[str] = None,
) -> None:
    """Interactive CLI chat loop."""
    gen = Generator(model, tokenizer, device=device)
    history: List[str] = []

    print("=" * 60)
    print("Chat started. Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    if system_prompt:
        history.append(f"<|system|> {system_prompt}")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append(f"<|human|> {user_input}")
        prompt = "\n".join(history) + "\n<|assistant|>"

        print("\nAssistant: ", end="", flush=True)
        response_parts: List[str] = []
        for chunk in gen.stream(prompt, max_new_tokens=512, temperature=0.7):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)

        full_response = "".join(response_parts)
        history.append(f"<|assistant|> {full_response}")
        print()
