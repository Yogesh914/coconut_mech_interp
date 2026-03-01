# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coconut is the official implementation of "Training Large Language Models to Reason in a Continuous Latent Space" (Meta, arXiv:2412.06769). It trains LLMs to perform reasoning using continuous latent representations instead of discrete chain-of-thought tokens.

## Commands

### Training
```bash
torchrun --nnodes 1 --nproc_per_node <N_GPUS> run.py args/<config>.yaml
```

### Evaluation
Set `only_eval: True` and `load_model_path` in the eval YAML, then run the same torchrun command with the eval config.

### Data Preprocessing
- **GSM8K**: `bash preprocessing/gsm_icot.bash` (requires HuggingFace `datasets`)
- **ProntoQA**: `python preprocessing/prontoqa.py` (requires generating data from official ProntoQA repo first)
- **ProsQA**: Pre-included in `data/`

### Dependencies
```bash
pip install -r requirements.txt
```
Key deps: torch 2.5.1, transformers 4.46.2, wandb, numpy, datasets, tqdm.

## Architecture

Four core files with clear separation of concerns:

- **`coconut.py`** — `Coconut(nn.Module)` wraps a HuggingFace causal LM (GPT2 or Llama3). Implements multi-pass forward with latent token feedback: on each pass, hidden states from the previous pass replace latent token embeddings. Special tokens: `<|latent|>`, `<|start-latent|>`, `<|end-latent|>`. Max 8 latent tokens (`MAX_N_LATENT`).

- **`run.py`** — Training/eval orchestrator. Supports FSDP (Llama) and DDP distributed training. Implements multi-stage training where latent token count increases progressively across stages. Four training modes: `coconut`, `cot`, `no_thoughts`, `no_cot` (set via YAML config flags).

- **`dataset.py`** — Loads JSON data (`question`/`answer`/`steps` format). `get_cot_latent_dataset()` builds training data with progressive latent replacement. `get_question_latent_dataset()` builds inference data. `MyCollator` handles padding with KV cache reuse optimization.

- **`utils.py`** — `Config` class (dict-to-object for YAML configs) and `set_seed()`.

### Configuration

YAML configs live in `args/`. Key parameters:
- Training mode flags: `coconut`, `cot`, `no_thoughts`, `no_cot` (mutually exclusive)
- `c_thought` — number of continuous thoughts per step
- `max_latent_stage` — maximum reasoning stages
- `epochs_per_stage` — epochs before increasing latent count
- `only_eval` — inference-only mode

### Training Flow

1. Load base model, add special latent tokens to tokenizer
2. Optional stage 0: CoT pre-training (used for GSM8K)
3. Stages 1+: progressively replace CoT steps with continuous latent tokens
4. Loss is masked on question and latent tokens (label_pad_token_id=-100)
5. Checkpoints saved per stage; preemption-aware resume supported
