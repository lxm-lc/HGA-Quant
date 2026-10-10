# VAR HGA Quantization

This directory contains the VAR-side generation and evaluation entry points for HGA quantization.

## Environment

Create the base generation environment with Python 3.10:

```bash
cd /path/to/HGA-mine/VAR
conda create -n var python=3.10 -y
conda activate var
pip install -r requirements.txt
pip install -e ../fast-hadamard-transform
```

## Downloads and Manual Paths

The one-click script does not download model checkpoints or evaluation assets automatically. Download the required files manually and place them at the paths shown below.

| Asset | Local path | Download page or source |
| --- | --- | --- |
| VAR-d30 checkpoint | `pretrained_models/var_d30.pth` | [Hugging Face file](https://huggingface.co/FoundationVision/var/blob/main/var_d30.pth) |
| VAR VAE checkpoint | `pretrained_models/vae_ch160v4096z32.pth` | [Hugging Face file](https://huggingface.co/FoundationVision/var/blob/main/vae_ch160v4096z32.pth) |
| ImageNet256 reference statistics | `evaluation/VIRTUAL_imagenet256_labeled.npz` | [OpenAI reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) |
| Inception graph | `classify_image_graph_def.pb` | [OpenAI Inception graph](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb) |

## Run Commands

The default configuration is VAR-d30 with HGA coarse-to-fine W4A6 quantization. It uses 4-bit weights, 4-bit low-precision activation stages, 8-bit high-precision activation stages, a configured average activation precision of 6 bits.

The shortest command is:

```bash
cd /path/to/HGA-mine/VAR
conda activate var
bash scripts/run_hga_var.sh
```

The same default configuration written explicitly is:

```bash
DEPTH=30 \
VAR_CKPT=pretrained_models/var_d30.pth \
VAE_CKPT=pretrained_models/vae_ch160v4096z32.pth \
QUANTIZE_WEIGHT=1 \
QUANTIZE_ACT=1 \
WEIGHT_BITS=4 \
MIXED_ACT_MODE=coarse_to_fine \
CTF_ACT_POLICY=early_high \
CTF_LOW_BITS=4 \
CTF_HIGH_BITS=8 \
CTF_TARGET_AVG_BITS=6.0 \
bash scripts/run_hga_var.sh
```

The script samples 50 images for each of the 1,000 ImageNet classes, producing 50,000 256x256 PNG files. A typical output directory is `output/VAR_HGA_W4A6_ctf_low4high8_YYYYMMDDHHMM/`. Generation runs first, and evaluation runs afterward only when `REFERENCE_NPZ` or `EVAL_SCRIPT` is provided. When `evaluation/VIRTUAL_imagenet256_labeled.npz` is present, the wrapper automatically uses it as `REFERENCE_NPZ`; evaluation also requires `classify_image_graph_def.pb` or an explicit `INCEPTION_PB` path.

The built-in evaluator reports FID, sFID, and Inception Score. To generate and evaluate in one command with explicit evaluation assets, use:

```bash
REFERENCE_NPZ=evaluation/VIRTUAL_imagenet256_labeled.npz \
INCEPTION_PB=evaluation/classify_image_graph_def.pb \
bash scripts/run_hga_var.sh
```

The shortest command for uniform W4A6 activation quantization is:

```bash
DEPTH=30 \
VAR_CKPT=pretrained_models/var_d30.pth \
MIXED_ACT_MODE=default \
WEIGHT_BITS=4 \
ACT_BITS=6 \
USE_HADAMARD=1 \
USE_TBGR=1 \
bash scripts/run_hga_var.sh
```

To evaluate an existing generation directory without generating images again, pass the directory containing the completed PNG samples and metadata to `EXISTING_SAMPLE_DIR`, taking directory `VAR_HGA_W4A6_ctf_low4high8_202609232047` as an example:

```bash
EVAL_ONLY=1 \
EXISTING_SAMPLE_DIR=output/VAR_HGA_W4A6_ctf_low4high8_202609232047 \
REFERENCE_NPZ=evaluation/VIRTUAL_imagenet256_labeled.npz \
INCEPTION_PB=classify_image_graph_def.pb \
bash scripts/run_hga_var.sh
```
