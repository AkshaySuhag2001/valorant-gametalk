#!/usr/bin/env python
"""Reference QLoRA training script for the Valorant Game-Talk adapter.

This reconstructs the recorded Colab setup. The original notebook was not
available locally, so settings absent from the saved run summary are explicit
defaults here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_MODEL = "Qwen/Qwen3-4B"


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array or JSON Lines export and validate its outer shape."""
    raw_text = path.read_text(encoding="utf-8")
    try:
        records = json.loads(raw_text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if isinstance(records, dict):
        records = records.get("data", records.get("rows", records.get("examples")))
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty JSON list of rows.")
    for index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} is not an object.")
        if not isinstance(row.get("prompt"), list) or not isinstance(
            row.get("completion"), list
        ):
            raise ValueError(
                f"Row {index} must have structured prompt and completion lists."
            )
        for key in ("prompt", "completion"):
            for message in row[key]:
                if not isinstance(message, dict) or not {
                    "role",
                    "content",
                }.issubset(message):
                    raise ValueError(
                        f"Row {index} has an invalid {key} message; "
                        "each message needs role and content."
                    )
    return records


def tokenize_rows(
    rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize chats and mask prompt tokens so loss is calculated on answers."""
    features: list[dict[str, list[int]]] = []
    for index, row in enumerate(rows):
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = tokenizer.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat-template prefix mismatch in row {index}; "
                "review the tokenizer template before training."
            )
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(f"Row {index} has no assistant target tokens.")
        if len(full_ids) > max_length:
            raise ValueError(
                f"Row {index} has {len(full_ids)} tokens, over max length "
                f"{max_length}. Increase the limit or shorten the example."
            )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        features.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return Dataset.from_list(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/training_rows.json"),
        help="Local JSON/JSONL export with structured prompt/completion rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_output/valorant-qwen3-4b-lora"),
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this 4-bit QLoRA run. Select a GPU runtime."
        )
    if not args.train_file.is_file():
        raise FileNotFoundError(
            f"Training data not found: {args.train_file}. "
            "See data/README.md for the expected JSON format."
        )

    set_seed(args.seed)
    rows = load_rows(args.train_file)
    print(f"Training rows: {len(rows)}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {args.base_model}")
    print(f"Adapter output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Tesla T4 supports FP16. This avoids the BF16 gradient-scaler issue seen
    # in earlier attempts in the Colab session.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Keep trainable adapters in FP32 while the frozen base uses 4-bit/FP16.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.to(torch.float32)
    model.print_trainable_parameters()

    train_dataset = tokenize_rows(rows, tokenizer, args.max_length)
    lengths = [len(item["input_ids"]) for item in train_dataset]
    print(f"Tokenized rows: {len(train_dataset)}")
    print(f"Longest sequence: {max(lengths)} tokens")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        warmup_steps=5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_steps=8,
        save_strategy="no",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    result = trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metrics = {
        key: float(value) if isinstance(value, (int, float)) else str(value)
        for key, value in result.metrics.items()
    }
    metrics["training_rows"] = len(rows)
    metrics["base_model"] = args.base_model
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Training complete. Adapter saved to {args.output_dir}")
    print(f"Training loss: {metrics.get('train_loss', 'not reported')}")


if __name__ == "__main__":
    main()
