#!/usr/bin/env python
"""Compare the base, prompted base, and fine-tuned adapter on fixed questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_ADAPTER = "AkshaySuhag/valorant-qwen3-4b-game-talk"
STYLE_SYSTEM_PROMPT = """\
Answer everyday questions accurately in exactly 2 concise sentences.
Sentence 1 gives practical advice. Sentence 2 adds one relevant Valorant
analogy using a specific term in its normal meaning. Prefer:
communication = comms or callout; practice = practice range; budgeting =
economy or eco round; changing a route = rotate; preparation = loadout;
learning from a mistake = replay review; a focused work block = round
objective. Choose only a term that fits; never list jargon, invent a term,
or use another game's language. Keep real-world advice literal.
For direct Valorant questions, answer factually and do not force an analogy.
No emojis.
"""


def read_questions(path: Path) -> list[str]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("questions")
    if not isinstance(value, list) or not value:
        raise ValueError("Question file must be a non-empty JSON array.")
    questions = [
        item["question"] if isinstance(item, dict) else item for item in value
    ]
    if not all(isinstance(question, str) and question.strip() for question in questions):
        raise ValueError("Every question must be a non-empty string.")
    return questions


def load_quantized_base(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=config,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def answer(
    model: Any,
    tokenizer: Any,
    question: str,
    system_prompt: str | None,
    max_new_tokens: int,
    repetition_penalty: float,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
    )
    input_device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(input_device)
    attention_mask = torch.ones_like(input_ids)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer_ids = generated[0, input_ids.shape[1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/holdout_questions.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/holdout_eval.json"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=110)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 4-bit evaluation run.")
    questions = read_questions(args.questions)
    model, tokenizer = load_quantized_base(args.base_model)

    results = []
    for index, question in enumerate(questions, start=1):
        base_answer = answer(
            model,
            tokenizer,
            question,
            system_prompt=None,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        prompted_answer = answer(
            model,
            tokenizer,
            question,
            system_prompt=STYLE_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        results.append(
            {
                "question": question,
                "base": base_answer,
                "prompted": prompted_answer,
            }
        )
        print(f"Base comparisons {index}/{len(questions)}")

    adapter_model = PeftModel.from_pretrained(model, args.adapter)
    adapter_model.eval()
    for index, row in enumerate(results, start=1):
        row["finetuned"] = answer(
            adapter_model,
            tokenizer,
            row["question"],
            system_prompt=STYLE_SYSTEM_PROMPT,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"Adapter answers {index}/{len(results)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
