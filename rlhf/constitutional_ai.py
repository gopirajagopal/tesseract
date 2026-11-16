"""Constitutional AI (CAI) — Anthropic's alignment technique.

CAI works in two stages:
  1. Supervised Learning (SL-CAI):
     a. Generate an initial response to a harmful/ambiguous prompt
     b. Ask the model to critique its response against a constitutional principle
     c. Ask the model to revise based on the critique
     d. Use the revised responses as SFT training data

  2. RL-CAI:
     - Use AI feedback (model scores own outputs against constitution)
       instead of human preferences to train the reward model
     - Then apply standard RLHF-PPO/DPO on top

Reference: https://arxiv.org/abs/2212.08073
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from inference.generate import Generator


CONSTITUTION = [
    "Choose the response that is least likely to contain harmful, unethical, or illegal content.",
    "Choose the response that is most helpful, honest, and harmless.",
    "Choose the response that would be least objectionable if seen by a child.",
    "Choose the response that most avoids encouraging violence, crime, or self-harm.",
    "Choose the response that is least likely to be embarrassing to a thoughtful person.",
    "Choose the response that avoids implying that humans are inferior or that AI is superior.",
    "Choose the response that most supports human autonomy and avoids being paternalistic.",
    "Choose the response that is most honest about being an AI and its limitations.",
    "Choose the response that best avoids discriminatory or stereotyping content.",
    "Choose the response that most clearly respects privacy and personal boundaries.",
]

CRITIQUE_PROMPT_TEMPLATE = """\
<|human|> {prompt}

<|assistant|> {initial_response}

<|human|> Please critique the above response. Identify any ways it might be harmful, \
unethical, dangerous, or dishonest. Consider the following principle:
{principle}

<|assistant|> Critique:"""

REVISION_PROMPT_TEMPLATE = """\
<|human|> {prompt}

<|assistant|> {initial_response}

<|human|> Critique: {critique}

Please revise your response to address the critique above. \
The revision should be helpful, harmless, and honest.

<|assistant|> Revised response:"""

AI_FEEDBACK_PROMPT_TEMPLATE = """\
Consider the following principle:
{principle}

Prompt: {prompt}

Response A: {response_a}
Response B: {response_b}

Which response better follows the principle above?
Answer with only "A" or "B".

<|assistant|>"""


class ConstitutionalAI:
    """CAI pipeline for self-improving alignment.

    Given a base/SFT model and a tokenizer, this class:
    1. Generates critiques of initial responses
    2. Generates revised responses
    3. Optionally generates AI feedback for RM training

    The resulting (prompt, revised_response) pairs can be used for
    further SFT, and the AI feedback pairs can train the reward model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config,
        constitution: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.constitution = constitution or CONSTITUTION
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(model, tokenizer, device=self.device)

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate a single response."""
        return self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    def critique_and_revise(
        self,
        prompt: str,
        initial_response: Optional[str] = None,
        num_revisions: Optional[int] = None,
    ) -> Dict[str, str]:
        """Run the full critique-revise loop for a single prompt.

        Returns:
            Dict with keys: initial_response, critique, revised_response
        """
        import random
        num_revisions = num_revisions or self.cfg.num_cai_revisions

        # Step 1: Generate initial response if not provided
        if initial_response is None:
            initial_response = self._generate(
                f"<|human|> {prompt}\n<|assistant|>",
                max_new_tokens=512,
            )

        current_response = initial_response
        critiques = []

        for i in range(num_revisions):
            # Pick a random constitutional principle
            principle = random.choice(self.constitution)

            # Step 2: Critique
            critique_prompt = CRITIQUE_PROMPT_TEMPLATE.format(
                prompt=prompt,
                initial_response=current_response,
                principle=principle,
            )
            critique = self._generate(critique_prompt, max_new_tokens=256)
            critiques.append(critique)

            # Step 3: Revise
            revision_prompt = REVISION_PROMPT_TEMPLATE.format(
                prompt=prompt,
                initial_response=current_response,
                critique=critique,
            )
            current_response = self._generate(revision_prompt, max_new_tokens=512)

        return {
            "prompt":           prompt,
            "initial_response": initial_response,
            "critique":         critiques[-1] if critiques else "",
            "revised_response": current_response,
        }

    def process_batch(
        self,
        prompts: List[str],
        verbose: bool = True,
    ) -> List[Dict[str, str]]:
        """Run CAI on a batch of prompts."""
        results = []
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"  CAI [{i+1}/{len(prompts)}]: {prompt[:60]}...")
            result = self.critique_and_revise(prompt)
            results.append(result)
        return results

    def generate_ai_feedback(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        principle: Optional[str] = None,
    ) -> str:
        """Use the model to pick which of two responses is better (AI feedback).

        Returns "A" or "B".
        """
        import random
        if principle is None:
            principle = random.choice(self.constitution)

        feedback_prompt = AI_FEEDBACK_PROMPT_TEMPLATE.format(
            principle=principle,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
        )
        answer = self._generate(feedback_prompt, max_new_tokens=5).strip().upper()
        return "A" if "A" in answer else "B"

    def build_preference_dataset(
        self,
        prompts: List[str],
        verbose: bool = True,
    ) -> List[Dict[str, str]]:
        """Build a preference dataset using AI feedback for RM training.

        For each prompt:
        1. Generate two candidate responses (at different temperatures)
        2. Use AI feedback to determine which is preferred
        3. Emit (prompt, chosen, rejected) triple

        Returns:
            List of dicts with keys: prompt, chosen, rejected
        """
        dataset = []
        for i, prompt in enumerate(prompts):
            if verbose:
                print(f"  AI Feedback [{i+1}/{len(prompts)}]...")

            # Two candidate responses
            resp_a = self._generate(f"<|human|> {prompt}\n<|assistant|>", max_new_tokens=512)
            resp_b = self._generate(f"<|human|> {prompt}\n<|assistant|>", max_new_tokens=512)

            winner = self.generate_ai_feedback(prompt, resp_a, resp_b)

            chosen   = resp_a if winner == "A" else resp_b
            rejected = resp_b if winner == "A" else resp_a

            dataset.append({
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
            })
        return dataset

    def save_dataset(self, dataset: List[Dict], path: str) -> None:
        """Save a generated dataset to JSONL."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(dataset)} examples → {path}")
