# ATML PA3 (EE-5102 / CS-6304, Fall 2026)

Code, configurations, split IDs and machine-readable results for Programming Assignment 3.

| Task | Folder | Status |
|---|---|---|
| 1. LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA-16) | `task1/` | done |
| 2. Vision-language model and visual grounding | `task2/` | done |
| 3. Test-time compute | `task3/` | done |

## Task 1: Large Language Model Fine-Tuning

Qwen3-0.6B (thinking disabled) is fine-tuned for one epoch on the first 40,000 eligible MetaMathQA examples with five methods. Each run is evaluated with lm-evaluation-harness on GSM8K, HellaSwag, BoolQ and ARC-Easy, and on 24 fixed qualitative prompts.

### Results

Accuracy (%) on the full test/validation sets. GSM8K uses flexible-extract exact match; HellaSwag is length-normalized; truncation is the share of GSM8K responses that hit the 1,024-token limit.

| Condition | GSM8K | HellaSwag | BoolQ | ARC-Easy | GSM8K truncated | Mean response tokens |
|---|---|---|---|---|---|---|
| Baseline | 43.4 | 47.3 | 64.4 | 60.8 | 1.1% | 227 |
| Full FT (update 625) | 55.0 | 45.8 | 61.2 | 66.1 | 0.9% | 143 |
| Full FT | 63.4 | 46.0 | 59.5 | 66.4 | 0.8% | 140 |
| L2-SP | 64.2 | 46.5 | 64.6 | 66.7 | 0.4% | 138 |
| LoRA r=4 | 60.3 | 47.3 | 69.8 | 65.8 | 0.7% | 141 |
| LoRA r=16 | 61.7 | 47.2 | 71.6 | 65.9 | 0.6% | 139 |
| OPLoRA r=16 | 62.9 | 47.3 | 70.8 | 65.9 | 0.5% | 139 |

Per-condition files are in `results/task1/eval/<condition>/` (`scores.json`, per-example rows, `qualitative.json`, and `merge_check.json` for adapters). Training curves and summaries are in `results/task1/training/`, and the `down_proj` diagnostics in `results/task1/diagnostics/`.

### Layout

| File | Purpose |
|---|---|
| `task1/config.py` | All fixed settings from the manual (seed 6304, batch 2x16, 1,250 updates, learning rates, LoRA/OPLoRA ranks) |
| `task1/data.py` | MetaMathQA filtering (empty, >2,048 tokens, GSM8K-test overlap after normalization), tokenization with response-only labels, probe-set sampling |
| `task1/lora.py` | LoRA and OPLoRA adapters (projected factors `B - U_k U_k^T B`, `A - A V_k V_k^T`), export, merging |
| `task1/train.py` | FP16 AMP training loop with FP32 master weights or FP32 adapters, L2-SP penalty, resumable checkpoints at updates 312/625/937/1250 |
| `task1/evaluate.py` | lm-eval benchmarks (zero-shot; GSM8K with the non-thinking chat template, greedy, 1,024 tokens), per-example outputs, truncation flags, qualitative generations, merged-vs-unmerged adapter check |
| `task1/run_queue.py` | Runs train/eval jobs across GPUs with a wall-clock budget (Kaggle 12 h sessions) |
| `task1/plots.py` | Report figures (`results/task1/figures/`) and LaTeX tables (`results/task1/tables/`) from the committed results |
| `task1/diagnostics.py` | Weight-space (`rho_k`, input-side fraction, `eta`) and activation-ratio diagnostics on `down_proj` in layers 0, 13, 27 (CPU) |
| `kaggle/task1_runner.ipynb` | Thin Kaggle notebook: clone repo, restore previous outputs, run the queue |
| `kaggle/task1_finalize.ipynb` | Packs scores, per-example rows, logs and summaries (without weights) into one zip |
| `results/task1/` | Split IDs, scores, per-example outputs, training logs and summaries |

### Reproduce

```bash
pip install -r requirements.txt
python -m task1.data                       # builds the training cache and probe set
python -m task1.train --run full_ft        # full_ft | l2sp | lora4 | lora16 | oplora16
python -m task1.evaluate --condition baseline
```

On Kaggle (GPU T4 x2, internet enabled), run `kaggle/task1_runner.ipynb` with "Save & Run All". To continue an interrupted session, attach the previous version's output as an input and run again; finished jobs are skipped and unfinished runs resume from their last recovery checkpoint.

Set `ATML_SMOKE_TEST=1` to run a 4-update CPU smoke test of the full pipeline.

## Task 2: Vision-Language Model and Visual Grounding

Frozen SigLIP-base features (196 x 768, cached in FP16) feed a trainable two-layer MLP connector into frozen Qwen3-0.6B (thinking disabled). Training runs three stages on one model: caption alignment (connector only, 2,250 updates), visual instruction tuning on generated presence/counting pairs with caption replay (connector + LoRA r16 on q/k/v/o, 1,500 updates), and natural VQAv2 specialization with replay (750 updates). Grounding is measured on complementary pairs with correct, complementary-swap, mismatched and no-image inputs.

### Results

Final-test accuracy (%) under four image conditions (questions and references fixed). Grounding uses normalized exact match (64 pairs / 128 questions); VQAv2 uses local soft consensus accuracy against all ten answers (96 pairs / 192 questions). Brackets are 95% pair-bootstrap intervals (4,000 resamples).

| Run | Suite | Correct | Complementary swap | Mismatched | No image | Pair acc. (correct) | Correct - swap |
|---|---|---|---|---|---|---|---|
| 196 tokens | Grounding | 71.9 [63.3, 80.5] | 16.4 | 31.2 | 39.1 | 56.2 | +55.5 [40.6, 69.5] |
| 196 tokens | VQAv2 | 60.7 [54.3, 67.1] | 27.2 | 29.3 | 31.3 | 31.2 | +33.4 [24.1, 42.8] |
| 49 tokens | Grounding | 68.8 [59.4, 78.1] | 14.1 | 31.2 | 38.3 | 54.7 | +54.7 [39.8, 68.8] |
| 49 tokens | VQAv2 | 56.1 [49.9, 62.4] | 29.4 | 30.6 | 32.3 | 22.9 | +26.7 [17.4, 35.8] |

Training took 1.86 h (196 tokens) and 0.78 h (49 tokens) on one T4 each; feature caching took 0.08 h. Full metrics, per-example rows and qualitative examples are in `results/task2/eval/<run>/`, learning curves and dev metrics in `results/task2/training/<run>/`, and the selected image/question/pair IDs in `results/task2/splits/`.

### Layout

| File | Purpose |
|---|---|
| `task2/config.py` | All fixed settings from the manual (seed 6304, split sizes, stage budgets and learning rates, token caps) |
| `task2/data.py` | Downloads COCO 2014 / VQAv2 annotations; selects captions, generated presence/counting pairs (with label-quality filters) and VQAv2 complementary pairs (consensus >= 8/10, differing answers); image-disjoint dev/test splits; image download |
| `task2/features.py` | Deterministic preprocessing and FP16 SigLIP feature cache (post-layernorm applied once, checked) |
| `task2/model.py` | Connector, 2 x 2 pooling for the 49-token variant, chat-format sequence building, example-weighted loss, generation |
| `task2/train.py` | Curriculum sampler, per-stage warmup + cosine schedule, FP16 autocast with FP32 trainable weights, connector calibration, resumable checkpoints every 250 updates, dev evaluation every 500 updates and at stage ends |
| `task2/evaluate.py` | Final-test generation and reference NLL under four image conditions, VQA soft accuracy, pair metrics, pair-bootstrap intervals, qualitative examples |
| `task2/vqa_eval.py` | VQA answer normalization and soft-consensus score |
| `task2/smoke.py` | Synthetic manifest for a CPU smoke test (`ATML_SMOKE_TEST=1`) |
| `kaggle/task2_data.ipynb` | CPU notebook: data selection and image download |
| `kaggle/task2_runner.ipynb` | GPU notebook: feature cache, then the 196-token run (GPU 0) and the optional 49-token run (GPU 1) |

### Reproduce

```bash
python -m task2.data                    # selections + images
python -m task2.features                # SigLIP feature cache
python -m task2.train --run vlm196      # vlm196 | vlm49
python -m task2.evaluate --run vlm196
```

## Task 3: Test-Time Compute

Offline experiments use the supplied ATML-PA3-Data rollouts (32 per question; Qwen2.5-3B-Instruct on a 1,000-question MATH subset, Qwen2.5-VL-3B-Instruct on MathVista testmini). Questions are split 200 development / 800 held-out with `default_rng(6304)`; ten seeded rollout orders are shared by all methods and stopping policies. Answers are grouped without gold labels (math-verify 0.9.0 for MATH; MathVista's answer-type, choice and precision rules, with options as letters) and graded with the same rules. A small live experiment compares a long response, parallel sampling and sequential refinement on 50 held-out MATH questions.

### Results

Held-out accuracy (%) at N = 32 candidates; brackets are 95% paired question bootstrap intervals (1,000 resamples) of the difference to self-consistency.

| Method | MATH | MathVista |
|---|---|---|
| FirstAns | 64.9 | 53.2 |
| Self-consistency (SC) | 73.8 | 61.8 |
| Max self-certainty | 70.6 (-3.2 [-5.6, -0.8]) | 54.8 (-7.0 [-10.1, -3.7]) |
| Self-certainty + RWC | 75.6 (+1.8 [0.7, 2.8]) | 62.5 (+0.7 [-0.6, 2.1]) |
| VisConf + RWC | - | 63.2 (+1.5 [0.1, 3.0]) |
| Oracle coverage | 91.4 | 89.5 |

Adaptive stopping (thresholds chosen on the development split) on held-out questions:

| Dataset | Policy | Accuracy | Tokens / question | Mean rollouts |
|---|---|---|---|---|
| MATH | Fixed-32 SC | 73.8 | 21,347 | 32 |
| MATH | Agreement (tau 0.5) | 72.6 | 6,697 | 7.9 |
| MATH | Quality-aware (tau 0.5, gamma 0) | 73.3 | 7,489 | 8.8 |
| MathVista | Fixed-32 SC | 61.8 | 8,898 | 32 |
| MathVista | Agreement (tau 0.625) | 61.9 | 4,689 | 14.0 |
| MathVista | Quality-aware (tau 0.625, gamma 0) | 62.0 | 4,755 | 14.2 |

Live comparison (50 MATH questions, 3,072-token output allowance, FP16 on one T4 per shard):

| Strategy | Accuracy | Output tokens | Time (s) | Truncated |
|---|---|---|---|---|
| Single 3,072-token response | 62 | 760 | 38.3 | 4% |
| Parallel 3 x 1,024 + SC | 54 | 1,857 | 92.3 | 17% per response |
| Initial 1,024 + revision 2,048 | 58 | 1,238 | 62.3 | 20% initial / 2% revision |

Refinement changed 1 wrong answer to correct and no correct answer to wrong. Full results, per-question rows, selected answers and split/order IDs are in `results/task3/`; figures and LaTeX tables in `results/task3/figures/` and `results/task3/tables/`.

### Layout

| File | Purpose |
|---|---|
| `task3/config.py` | Fixed settings (seed 6304, splits, budgets, calibration percentiles, stopping grids, live-generation settings) |
| `task3/answers.py` | Balanced-brace `\boxed{}` extraction, math-verify grouping and grading, MathVista normalization |
| `task3/data.py` | Joins the three tables on (dataset, sample_id, rollout_id), checks token-list alignment, builds groups, splits and orders |
| `task3/selection.py` | Self-consistency, highest-score selection, rank-weighted consensus, candidate-relative VisConf calibration, stopping evidence |
| `task3/analysis.py` | Steps 1, 2, 3 and 5: fixed budgets, failure decomposition, signal analysis, adaptive stopping, bootstrap intervals |
| `task3/live.py` | Step 4 live generation (two shards) and grading |
| `task3/plots.py` | Report figures and tables |
| `kaggle/task3_offline.ipynb`, `kaggle/task3_live.ipynb` | Kaggle notebooks (CPU offline analysis; GPU T4 x2 live generation) |

### Reproduce

```bash
python -m task3.data            # with ATML3_DATA pointing at the downloaded ATML-PA3-Data folder
python -m task3.analysis
python -m task3.live --shard 0 --num-shards 2 && python -m task3.live --shard 1 --num-shards 2
python -m task3.live --summarize
python -m task3.plots
```

## External resources

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) task definitions (`gsm8k_cot`, `hellaswag`, `boolq`, `arc_easy`), used unmodified.
- `task1/data/first_plot_questions.yaml` is copied from the [emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) repository (Betley et al.).
- `task2/vqa_eval.py` follows the answer normalization of the official [VQA evaluation code](https://github.com/GT-Vision-Lab/VQA) (punctuation, number words, articles, contractions).
- COCO 2014 captions/instances and VQAv2 questions, annotations and complementary pairs are downloaded from their official hosts.
- Task 3 uses [math-verify](https://github.com/huggingface/Math-Verify) for MATH equivalence; the MathVista normalization follows `calculate_score.py` in the [MathVista repository](https://github.com/lupantech/MathVista), and its `testmini.json` metadata is downloaded at run time.
- LoRA (Hu et al.) and OPLoRA (Xiong and Xie) are implemented from the papers' equations; no adapter library is used.
