# HART HGA Quant Evaluation

This directory contains the HART-side generation and evaluation entry points for HGA quantization.

## Environment

```bash
cd /path/to/HGA-mine/HART
conda create -n hart python=3.10
conda activate hart
conda install -c nvidia cuda-toolkit -y
pip install -e .
cd hart/kernels && python setup.py install

cd /path/to/HGA-mine/HART
pip install -e ../fast-hadamard-transform
```

## Downloads and Manual Paths

Large checkpoints and benchmark assets are not part of the Git source tree. Place them at these default paths relative to `HART/`.

| Asset | Default path | Download page or source |
| --- | --- | --- |
| HART checkpoint | `checkpoints/hart-0.7b-1024px/llm` | [mit-han-lab/hart-0.7b-1024px](https://huggingface.co/mit-han-lab/hart-0.7b-1024px) |
| Qwen2-VL | `checkpoints/Qwen2-VL-1.5B-Instruct` | [mit-han-lab/Qwen2-VL-1.5B-Instruct](https://huggingface.co/mit-han-lab/Qwen2-VL-1.5B-Instruct) |
| ShieldGemma | `checkpoints/shieldgemma-2b` | [google/shieldgemma-2b](https://huggingface.co/google/shieldgemma-2b) |
| Shared Hadamard | `../fast-hadamard-transform` | [Dao-AILab/fast-hadamard-transform](https://github.com/Dao-AILab/fast-hadamard-transform) |
| DPG-Bench files | `checkpoints/dpg_bench/` | [DPG-Bench source files](https://github.com/TencentQQGYLab/ELLA) |
| MJHQ-30K metadata and images | `checkpoints/MJHQ-30K/` | [playgroundai/MJHQ-30K](https://huggingface.co/datasets/playgroundai/MJHQ-30K) |

The DPG workflow expects the DPG-Bench files directly under `checkpoints/dpg_bench/`:

```text
checkpoints/dpg_bench/
+-- prompts/
+-- dpg_bench.csv
+-- compute_dpg_bench.py
+-- dist_eval.sh
```

The MJHQ workflow expects the official metadata object and reference images under `checkpoints/MJHQ-30K/`:

```text
checkpoints/MJHQ-30K/
+-- meta_data.json
+-- images/
```

If a model checkpoint is stored somewhere else, pass its path explicitly:

```bash
MODEL_PATH=/path/to/hart-0.7b-1024px/llm \
TEXT_MODEL_PATH=/path/to/Qwen2-VL-1.5B-Instruct \
bash scripts/run_dpg.sh
```

The DPG evaluator additionally depends on ModelScope's mPLUG VQA model (`damo/mplug_visual-question-answering_coco_large_en`), which is fetched and cached by ModelScope at runtime.

## Run Commands

DPG-Bench and MJHQ are the maintained one-click benchmarks. The default setting is HGA-Quant coarse-to-fine W4A6: 4-bit weights, activation precision switching between 4 and 8 bits, target average activation precision 6 bits, Hadamard rotation enabled, T-BGR enabled, and evaluation enabled.

```bash
bash scripts/run_dpg.sh
bash scripts/run_mjhq.sh
```

The full DPG command with the default HGA configuration written out is:

```bash
QUANTIZE_WEIGHT=1 \
QUANTIZE_ACT=1 \
WEIGHT_BITS=4 \
MIXED_ACT_MODE=coarse_to_fine \
CTF_ACT_POLICY=early_high \
CTF_LOW_BITS=4 \
CTF_HIGH_BITS=8 \
CTF_TARGET_AVG_BITS=6.0 \
bash scripts/run_dpg.sh
```

Uniform W4A6 uses 6-bit activations without the coarse-to-fine scheduler:

```bash
QUANTIZE_WEIGHT=1 \
QUANTIZE_ACT=1 \
WEIGHT_BITS=4 \
ACT_BITS=6 \
MIXED_ACT_MODE=default \
bash scripts/run_dpg.sh
```

To evaluate an existing DPG run without generating images again, taking directory `HART_HGA_W4A6_ctf_low4high8_202609232047` as an example:

```bash
EVAL_ONLY=1 \
EXISTING_DPG_RUN_DIR=output/HART_HGA_W4A6_ctf_low4high8_202609232047 \
bash scripts/run_dpg.sh
```

Switch the last line to `bash scripts/run_mjhq.sh` to run the same settings on MJHQ.

Set `DPG_LIMIT` or `MJHQ_LIMIT` for smoke runs. Set `RUN_EVAL=0` only when MJHQ image generation is needed without evaluation.
