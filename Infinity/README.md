# Infinity HGA Quant Evaluation

This directory contains the Infinity-side generation and evaluation entry points for HGA quantization.

## Environment

Create the base generation environment with Python 3.10:

```bash
cd /path/to/HGA-mine/Infinity
conda create -n inf python=3.10 -y
conda activate inf
pip install -r requirements.txt
pip install -e ../fast-hadamard-transform
```

The `fast-hadamard-transform` directory should be placed at the workspace root, next to `Infinity`. It provides the `fast_hadamard_transform` package imported by `infinity/models/quant.py`.

Clone the generation environment for the three evaluators, then install evaluator-specific packages:

```bash
conda create -n inf_ir --clone inf
conda create -n inf_ge --clone inf
conda create -n inf_dpg --clone inf

conda activate inf_ir
pip install -r requirements-imagereward.txt

conda activate inf_ge
pip install -r requirements-geneval.txt
mim install mmengine "mmcv-full==1.7.2"

conda activate inf_dpg
pip install -r requirements-dpg.txt
```

The scripts support separate generation and evaluation environments. For example:

```bash
PYTHON_CONDA_ENV=[gen_ir] EVAL_CONDA_ENV=[eval_ir] bash scripts/run_imagereward.sh
PYTHON_CONDA_ENV=[gen_ge] EVAL_CONDA_ENV=[eval_ge] bash scripts/run_geneval.sh
PYTHON_CONDA_ENV=[gen_dpg] EVAL_CONDA_ENV=[eval_dpg] bash scripts/run_dpg.sh
```

## Downloads and Manual Paths

The three Infinity text-to-image assets are downloaded automatically when their default paths are missing. The GenEval detector checkpoint is not downloaded by the one-click scripts and must be placed manually.

| Asset | Default path | Download page or source |
| --- | --- | --- |
| Infinity-2B checkpoint | `weights/infinity_2b_reg.pth` | [Hugging Face file](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_2b_reg.pth) |
| Infinity VAE | `weights/infinity_vae_d32reg.pth` | [Hugging Face file](https://huggingface.co/FoundationVision/Infinity/blob/main/infinity_vae_d32reg.pth) |
| FLAN-T5-XL | `weights/flan-t5-xl/` | [Hugging Face model page](https://huggingface.co/google/flan-t5-xl) |
| GenEval Mask2Former checkpoint | `evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth` | [Hugging Face file](https://huggingface.co/tsbpp/geneval_mask2former/blob/main/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth) |

If an asset is stored somewhere else, pass its path explicitly. The scripts only auto-download when the default path is used:

```bash
MODEL_PATH=/path/to/infinity_2b_reg.pth \
VAE_PATH=/path/to/infinity_vae_d32reg.pth \
TEXT_ENCODER_CKPT=/path/to/flan-t5-xl \
bash scripts/run_imagereward.sh
```

For GenEval, the checkpoint filename must match `DETECTOR_NAME` inside `DETECTOR_MODEL_DIR`:

```bash
DETECTOR_NAME=mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco \
DETECTOR_MODEL_DIR=/path/to/mask2former \
DETECTOR_CONFIG=/path/to/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
bash scripts/run_geneval.sh
```

## Run Commands

ImageReward is the representative command. With the generation environment activated, the shortest form for our HGA-Quant coarse-to-fine W4A6 (weights use 4-bit quantization, activation precision switches between 4 and 8 bits, and the configured average activation precision is 6 bits) is:

```bash
bash scripts/run_imagereward.sh
```

You can run it with explicit configuration as well:

```bash
PYTHON_CONDA_ENV=inf_ir EVAL_CONDA_ENV=inf_ir MIXED_ACT_MODE=coarse_to_fine CTF_ACT_POLICY=early_high CTF_LOW_BITS=4 CTF_HIGH_BITS=8 CTF_TARGET_AVG_BITS=6.0 bash scripts/run_imagereward.sh
```

The shortest form for uniform W4A6 quantization is:

```bash
PYTHON_CONDA_ENV=inf_ir EVAL_CONDA_ENV=inf_ir MIXED_ACT_MODE=default WEIGHT_BITS=4 ACT_BITS=6 bash scripts/run_imagereward.sh
```

To evaluate an existing generation directory without generating images again:

```bash
EVAL_ONLY=1 \
EXISTING_RUN_DIR=output/Infinity_HGA_W4A6_ctf_low4high8_202609232047 \
PYTHON_CONDA_ENV=inf_ir \
EVAL_CONDA_ENV=inf_ir \
bash scripts/run_imagereward.sh
```

For DPG or GenEval, replace `run_imagereward.sh` and the evaluator environment with the corresponding script and environment.

Each new run is recorded in `output/run_records.json` and `<run_dir>/run_config.json`. Its status is `in_progress` while the run is active and `completed` after the generation and evaluation steps finish successfully.

## Notes

Infinity GenEval results use the rewritten prompt set in `Infinity/evaluation/gen_eval/prompts/evaluation_metadata.jsonl`. This set is based on the original GenEval prompt templates and was rewritten by humans using GPT, rather than using the `PromptRewriter` included with the project itself. The evaluation metadata and scoring protocol remain GenEval-style, and these prompts were used consistently throughout the experiments.
