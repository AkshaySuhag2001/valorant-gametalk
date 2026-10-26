# Valorant Game-Talk: Qwen3-4B Fine-Tuning

A small QLoRA fine-tuning project exploring whether a language model can answer everyday questions clearly while using a light, context-appropriate Valorant voice.

**Model adapter:** [AkshaySuhag/valorant-qwen3-4b-game-talk](https://huggingface.co/AkshaySuhag/valorant-qwen3-4b-game-talk)  
**Full evaluation report:** [Download the PDF](reports/valorant_gametalk_evaluation_report.pdf)

## Project at a glance

- **Base model:** Qwen/Qwen3-4B
- **Training method:** 4-bit NF4 QLoRA with LoRA rank 16, alpha 32, and dropout 0.05
- **Training set:** 65 examples (49 everyday questions and 16 Valorant term explainers)
- **Compute:** Tesla T4 GPU
- **Training run:** 6 epochs, 102 optimizer steps, about 4 minutes 23 seconds
- **Trainable parameters:** 33,030,144 of 4,055,498,240 (0.8145%)
- **Mean training loss:** 1.445

The published Hugging Face repository contains the adapter; this repository provides the project summary and full comparison report. It does not include the training notebook or training data.

## Code

The `scripts/` directory contains a reference QLoRA training pipeline and a three-condition evaluation runner. The Colab notebook itself was not available among the local project files, so these scripts reconstruct the recorded workflow rather than claim to be the original notebook cells.

The training script expects a local JSON file at `data/training_rows.json` by default. It should be a list of 65 records with structured `prompt` and `completion` message arrays. The dataset is intentionally git-ignored and is not part of this repository. See [the data schema](data/README.md).

To train in a CUDA environment:

    python scripts/train_qlora.py --train-file data/training_rows.json --output-dir training_output/valorant-qwen3-4b-lora

To generate a fresh three-way comparison:

    python scripts/evaluate.py --adapter AkshaySuhag/valorant-qwen3-4b-game-talk --questions data/holdout_questions.json --output results/holdout_eval.json

The evaluation script saves all three answer variants. It does not recompute the original 19-term exact-match score because the full historical term whitelist was not part of the available project files; the report documents the original measured result.

Install the pinned Python packages from `requirements.txt` in a CUDA environment with a compatible PyTorch build. The original run used Colab's preinstalled PyTorch 2.11.0+cu130 on a Tesla T4.

## Evaluation summary

Twelve newly written holdout questions were answered under three conditions: the base model, the base model with a Valorant style instruction, and the fine-tuned adapter with the same instruction. Generations used a shared 110-token cap.

| Condition | Answers with an exact term from the fixed 19-term list |
|---|---:|
| Base model | 0/12 |
| Base + style prompt | 0/12 |
| Adapter + same prompt | 8/12 |

The prompted baseline sometimes used other Valorant or gaming references that were not in the fixed list. Therefore, 0/12 is a whitelist count, not evidence that the prompted model used no game language. A single-pass qualitative review judged 9/12 adapter answers actionable and broadly sound; three were flagged for revision. The small evaluation does not establish a correctness improvement over prompting alone, and some game analogies were awkward.

The PDF includes all 12 questions with the base, prompted-base, and adapter answers, plus the evaluation method, findings, caveats, and next steps.

## Intended behavior

For ordinary questions, answers should be useful first and use a fitting Valorant reference sparingly. Direct questions about Valorant should receive factual answers without a forced analogy. Real-world instructions should remain literal and safe.

## Notes

- Training data contains no confidential or personal information.
- The term-match metric is a limited style indicator, not a measure of answer quality.
- This repository is a project showcase; adapter weights are hosted on Hugging Face.

## References

- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [Transformers Trainer documentation](https://huggingface.co/docs/transformers/trainer)
