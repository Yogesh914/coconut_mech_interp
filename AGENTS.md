# Repository Guidelines

## Project Structure & Module Organization
`run.py` is the distributed training and evaluation entrypoint. Core logic lives in `coconut.py` (model wrapper), `dataset.py` (dataset building and collation), and `utils.py` (lightweight config and seed helpers). Store run configs in `args/*.yaml`, preprocessing scripts in `preprocessing/`, prepared datasets in `data/`, and training artifacts in `ckpts/`. Keep exploratory work in `notebook.ipynb` or `experiments/` rather than adding ad hoc top-level scripts.

## Build, Test, and Development Commands

```bash
pip install -r requirements.txt
```

Prepare GSM data with `bash preprocessing/gsm_icot.bash`. Prepare ProntoQA after generating the raw file with `python preprocessing/prontoqa.py`. Launch training or evaluation with `torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml`; swap in another YAML from `args/` for CoT, ProntoQA, ProsQA, or eval runs. Log in to Weights & Biases with `wandb login` before non-debug runs.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes (`Coconut`, `Config`, `MyCollator`), and concise module names. Prefer explicit imports and small helper functions over deeply nested logic. Keep config filenames descriptive and dataset-specific, for example `prosqa_coconut_eval.yaml`.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Treat preprocessing and training smoke runs as the current validation path. Before opening a PR, run the narrowest meaningful check for your change, such as `python preprocessing/prontoqa.py` or a debug training pass with a target YAML. If you add testable utility logic, add a focused script or test file rather than relying only on manual notebook checks.

## Commit & Pull Request Guidelines
Match the repository’s history: short, imperative commit subjects such as `Update load_model_path in gsm_coconut.yaml` or `Delete wandb directory`. Keep commits scoped to one change. PRs should state the dataset/config affected, the exact command used to validate the change, and any GPU or checkpoint assumptions. Include metric deltas or sample outputs when behavior changes.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a folder name based on today's date (e.g. `mar5`). The branch `coconut_mech_interp/<tag>` must not already exist — this is a fresh run.
2. **Create the folder**
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context. and all the other files mentioned
4. **Confirm and go**: Confirm setup looks good.

The main checkpoint to use is at `/workspace/coconut_interp/checkpoint_23` which was a result from the run of `torchrun --nnodes 1 --nproc_per_node 4 run.py args/gsm_coconut.yaml`

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single MPS Apple MacBook.

**What you CAN do:**
- Modify only your own code and experiments in the experiments folder.

**What you CANNOT do:**
- Modify `coconut.py` or any other code in the current state of the repository, only add to it.
- Modify the evaluation harness. 

**The first run**: Your very first run should always be to just have an experiment running the model from the checkpoint and if implemented correctly you should see 33.4% as the test set accuracy.

**The goal is simple.** The goal is that you want to steer the COCONUT model to be able to increase accuracy significantly on GSM8K test set. The goal is to do novel mechanistic interpretability research on this model and discover something that has not been seen before, something big for a NeurIPS level research paper. I have also a summary of a paper doing mech interp research on coconut models that got into NeurIPS as well at `paper_summary.md`. This paper will give you an idea of everything done in the mech interp space with this model, DON'T FOLLOW UP OR COPY THE IDEA FROM THIS PAPER, YOU ARE EXPLORING IN OTHER SPACES OF MECHANISTIC INTERPRETABILITY. The model checkpoint you will be working with will be different as it is a GPT2 level sized model not 2 layers, and is trained on GSM8K instead. Explore ideas of vector steering to see if we can steer the model towards correct answers, and also explore if there is anything interesting in the mech interp side as the model does worse than regular CoT with the GSM8K dataset, maybe steering the model to think more/harder or steering the model to think more broadly than narrow. You want to steer the model to be able to increase accuracy significantly. You may also web search for further research and context about the field.

**Remember you have access to skills as well.**

## Logging results

When an experiment is done, log it to `findings.md` (here you log all your findings and keep a concise note and track of what you have experimented/tried already).

Make sure all experimental things fall under a folder called `experiments` and make sure to double check always the tokenizer and the implementations, make sure to use exactly the code provided for inference for this model, you may import the functions and use them directly, from `run.py` `coconut.py` `dataset.py` 

You are only allowed to run and code one `experiment.py` file, and will keep updating it.


## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `experiment.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run experiment.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If accuracy improved, you "advance" the branch, keeping the git commit
9. If accuracy is around equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 15 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~10 minutes then you can run approx 6/hour, for a total of about 50 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!