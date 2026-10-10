# FlexVAR HGA Quant Evaluation

This directory contains the FlexVAR-side generation and ImageNet C2I evaluation entry points for HGA quantization.

## Environment

Create the generation and evaluation environment with Python 3.10:

```bash
cd /path/to/HGA-mine/FlexVAR
conda create -n flexvar python=3.10 -y
conda activate flexvar
pip install -r requirements.txt
pip install -e ../fast-hadamard-transform
```

## Downloads and Manual Paths

Place the FlexVAR checkpoints and evaluation assets at the default paths below. The scripts do not auto-download these large files except for the Inception graph used by the evaluator.

| Asset | Default path | Download page or source |
| --- | --- | --- |
| FlexVAE checkpoint | `pretrained/FlexVAE.pth` | [Hugging Face file](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVAE.pth) |
| FlexVAR d16 checkpoint | `pretrained/FlexVARd16-epo179.pth` | [Hugging Face file](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVARd16-epo179.pth) |
| FlexVAR d20 checkpoint | `pretrained/FlexVARd20-epo249.pth` | [Hugging Face file](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVARd20-epo249.pth) |
| FlexVAR d24 checkpoint | `pretrained/FlexVARd24-epo349.pth` | [Hugging Face file](https://huggingface.co/jiaosiyu1999/FlexVAR/resolve/main/FlexVARd24-epo349.pth) |
| ImageNet256 reference statistics | `utils/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz` | [OpenAI reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) |
| Inception graph | `classify_image_graph_def.pb` | [OpenAI Inception graph](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb) |

The default checkpoint is selected from `DEPTH`:

```text
DEPTH=16 -> pretrained/FlexVARd16-epo179.pth
DEPTH=20 -> pretrained/FlexVARd20-epo249.pth
DEPTH=24 -> pretrained/FlexVARd24-epo349.pth
```

If an asset is stored somewhere else, pass its path explicitly:

```bash
DEPTH=24 \
VAR_CKPT=/path/to/FlexVARd24-epo349.pth \
VAE_CKPT=/path/to/FlexVAE.pth \
REFERENCE_NPZ=/path/to/VIRTUAL_imagenet256_labeled.npz \
bash scripts/run_hga_flexvar.sh
```

## Run Commands

With the environment activated, the shortest form for HGA-Quant coarse-to-fine W4A6 is:

```bash
bash scripts/run_hga_flexvar.sh
```

The default run uses `DEPTH=24`, `pretrained/FlexVARd24-epo349.pth`, 4-bit weights, coarse-to-fine activation quantization between 4 and 8 bits with configured average activation precision 6, Hadamard rotation, T-BGR outlier handling, 50,000 ImageNet-256 samples, and the built-in evaluator:

```text
utils/evaluations/c2i/evaluator.py
```

You can run the same configuration explicitly:

```bash
DEPTH=24 \
QUANTIZE_WEIGHT=1 \
QUANTIZE_ACT=1 \
WEIGHT_BITS=4 \
MIXED_ACT_MODE=coarse_to_fine \
CTF_ACT_POLICY=early_high \
CTF_LOW_BITS=4 \
CTF_HIGH_BITS=8 \
CTF_TARGET_AVG_BITS=6.0 \
bash scripts/run_hga_flexvar.sh
```

Uniform W4A6 uses 6-bit activations without the coarse-to-fine scheduler:

```bash
DEPTH=24 \
QUANTIZE_WEIGHT=1 \
QUANTIZE_ACT=1 \
WEIGHT_BITS=4 \
ACT_BITS=6 \
MIXED_ACT_MODE=default \
bash scripts/run_hga_flexvar.sh
```

To evaluate an existing generated NPZ without generating images again:

```bash
EVAL_ONLY=1 \
EXISTING_SAMPLE_NPZ=output/FlexVAR_HGA_W4A6_ctf_low4high8_202609232047.npz \
bash scripts/run_hga_flexvar.sh
```
