#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-}"
if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

EVAL_ONLY="${EVAL_ONLY:-0}"
EXISTING_DPG_RUN_DIR="${EXISTING_DPG_RUN_DIR:-}"
EXISTING_DPG_IMAGE_DIR="${EXISTING_DPG_IMAGE_DIR:-}"
EXISTING_DPG_MANIFEST_FILE="${EXISTING_DPG_MANIFEST_FILE:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"

DPG_PROMPT_DIR="${DPG_PROMPT_DIR:-$ROOT_DIR/checkpoints/dpg_bench/prompts}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints/hart-0.7b-1024px/llm}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-$ROOT_DIR/checkpoints/Qwen2-VL-1.5B-Instruct}"
DPG_CSV="${DPG_CSV:-$ROOT_DIR/checkpoints/dpg_bench/dpg_bench.csv}"
DPG_PROCESSES="${DPG_PROCESSES:-1}"
DPG_PORT="${DPG_PORT:-29500}"
ACCELERATE="${ACCELERATE:-accelerate}"

# These values are part of the original HART DPG protocol.
DPG_LIMIT="${DPG_LIMIT:-0}"
DPG_OFFSET="${DPG_OFFSET:-0}"
DPG_RESOLUTION=1024
SEED=1
USE_EMA=1
MAX_TOKEN_LENGTH=300
USE_LLM_SYSTEM_PROMPT=1
CFG=4.5
MORE_SMOOTH=1

QUANTIZE_WEIGHT="${QUANTIZE_WEIGHT:-1}"
QUANTIZE_ACT="${QUANTIZE_ACT:-1}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACT_BITS="${ACT_BITS:-6}"
USE_HADAMARD="${USE_HADAMARD:-1}"
USE_TBGR="${USE_TBGR:-1}"
TBGR_OUTLIER_FACTOR="${TBGR_OUTLIER_FACTOR:-8}"
TBGR_OUTLIER_SIGMA="${TBGR_OUTLIER_SIGMA:-3}"
MIXED_ACT_MODE="${MIXED_ACT_MODE:-coarse_to_fine}"
CTF_ACT_POLICY="${CTF_ACT_POLICY:-early_high}"
CTF_LOW_BITS="${CTF_LOW_BITS:-4}"
CTF_HIGH_BITS="${CTF_HIGH_BITS:-8}"
CTF_TARGET_AVG_BITS="${CTF_TARGET_AVG_BITS:-6.0}"
CTF_U_LATE_HIGH_COUNT="${CTF_U_LATE_HIGH_COUNT:-1}"

case "${MIXED_ACT_MODE,,}" in
  default)
    MIXED_ACT_MODE="default"
    ;;
  ctf|coarse_to_fine)
    MIXED_ACT_MODE="coarse_to_fine"
    ;;
  *)
    echo "MIXED_ACT_MODE must be default or ctf/coarse_to_fine." >&2
    exit 2
    ;;
esac
if [[ "$MIXED_ACT_MODE" == "coarse_to_fine" &&
      ( "$QUANTIZE_WEIGHT" != "1" || "$QUANTIZE_ACT" != "1" ) ]]; then
  echo "coarse_to_fine mode requires QUANTIZE_WEIGHT=1 and QUANTIZE_ACT=1." >&2
  exit 2
fi

format_number() {
  printf '%s' "$1" | sed 's/\.0*$//; s/\./p/g'
}

if [[ "$QUANTIZE_WEIGHT" == "1" ]]; then
  WEIGHT_TAG="$WEIGHT_BITS"
else
  WEIGHT_TAG="fp"
fi
if [[ "$QUANTIZE_ACT" == "1" ]]; then
  if [[ "$MIXED_ACT_MODE" == "coarse_to_fine" ]]; then
    ACT_TAG="$(format_number "$CTF_TARGET_AVG_BITS")"
    QUANT_TAG="W${WEIGHT_TAG}A${ACT_TAG}_ctf_low${CTF_LOW_BITS}high${CTF_HIGH_BITS}"
  else
    QUANT_TAG="W${WEIGHT_TAG}A${ACT_BITS}_default"
  fi
else
  QUANT_TAG="W${WEIGHT_TAG}Afp_default"
fi

if [[ "$EVAL_ONLY" == "1" ]]; then
  if [[ -z "$EXISTING_DPG_RUN_DIR" && -z "$EXISTING_DPG_IMAGE_DIR" ]]; then
    echo "EVAL_ONLY=1 requires EXISTING_DPG_RUN_DIR or EXISTING_DPG_IMAGE_DIR." >&2
    exit 2
  fi
  if [[ -n "$EXISTING_DPG_RUN_DIR" ]]; then
    RUN_DIR="$EXISTING_DPG_RUN_DIR"
    IMAGE_DIR="${EXISTING_DPG_IMAGE_DIR:-$RUN_DIR/images}"
  else
    IMAGE_DIR="$EXISTING_DPG_IMAGE_DIR"
    RUN_DIR="$(cd "$IMAGE_DIR/.." && pwd)"
  fi
else
  RUN_STEM="HART_HGA_${QUANT_TAG}_$(date +%Y%m%d%H%M)"
  RUN_DIR="$OUTPUT_ROOT/$RUN_STEM"
  RUN_INDEX=1
  while [[ -e "$RUN_DIR" ]]; do
    RUN_DIR="$OUTPUT_ROOT/${RUN_STEM}_${RUN_INDEX}"
    RUN_INDEX=$((RUN_INDEX + 1))
  done
  IMAGE_DIR="$RUN_DIR/images"
fi
MANIFEST_FILE="${EXISTING_DPG_MANIFEST_FILE:-$RUN_DIR/dpg_manifest.jsonl}"
RESULT_FILE="${DPG_RESULT_FILE:-$RUN_DIR/dpg_result.txt}"

if [[ "$EVAL_ONLY" == "1" ]]; then
  if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "Existing DPG image directory does not exist: $IMAGE_DIR" >&2
    exit 2
  fi
  echo "EVAL_ONLY=1; skip DPG model loading and image generation."
else
  mkdir -p "$IMAGE_DIR"

  "$PYTHON" - \
  "$DPG_PROMPT_DIR" \
  "$IMAGE_DIR" \
  "$MANIFEST_FILE" \
  "$DPG_LIMIT" \
  "$DPG_OFFSET" \
  "$DPG_RESOLUTION" \
  "$MODEL_PATH" \
  "$TEXT_MODEL_PATH" \
  "$SEED" \
  "$USE_EMA" \
  "$MAX_TOKEN_LENGTH" \
  "$USE_LLM_SYSTEM_PROMPT" \
  "$CFG" \
  "$MORE_SMOOTH" \
  "$QUANTIZE_WEIGHT" \
  "$QUANTIZE_ACT" \
  "$WEIGHT_BITS" \
  "$ACT_BITS" \
  "$USE_HADAMARD" \
  "$USE_TBGR" \
  "$TBGR_OUTLIER_FACTOR" \
  "$TBGR_OUTLIER_SIGMA" \
  "$MIXED_ACT_MODE" \
  "$CTF_ACT_POLICY" \
  "$CTF_LOW_BITS" \
  "$CTF_HIGH_BITS" \
  "$CTF_TARGET_AVG_BITS" \
  "$CTF_U_LATE_HIGH_COUNT" <<'PY'
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image
from sample import HARTSampler


(
    prompt_dir,
    image_dir,
    manifest_file,
    limit,
    offset,
    resolution,
    model_path,
    text_model_path,
    seed,
    use_ema,
    max_token_length,
    use_llm_system_prompt,
    cfg,
    more_smooth,
    quantize_weight,
    quantize_act,
    weight_bits,
    act_bits,
    use_hadamard,
    use_tbgr,
    tbgr_outlier_factor,
    tbgr_outlier_sigma,
    mixed_act_mode,
    ctf_act_policy,
    ctf_low_bits,
    ctf_high_bits,
    ctf_target_avg_bits,
    ctf_u_late_high_count,
) = sys.argv[1:]

prompt_dir = Path(prompt_dir)
image_dir = Path(image_dir)
manifest_file = Path(manifest_file)
limit = int(limit)
offset = int(offset)
resolution = int(resolution)
resampling = getattr(Image, "Resampling", Image)

prompt_files = sorted(prompt_dir.glob("*.txt"), key=lambda path: path.stem)
if offset:
    prompt_files = prompt_files[offset:]
if limit > 0:
    prompt_files = prompt_files[:limit]
if not prompt_files:
    raise SystemExit(f"No DPG prompt files found in {prompt_dir}")

sampler_args = Namespace(
    model_path=model_path,
    text_model_path=text_model_path,
    use_safety_checker=0,
    seed=int(seed),
    use_ema=int(use_ema),
    max_token_length=int(max_token_length),
    use_llm_system_prompt=int(use_llm_system_prompt),
    cfg=float(cfg),
    more_smooth=int(more_smooth),
    quantize_weight=int(quantize_weight),
    quantize_act=int(quantize_act),
    weight_bits=int(weight_bits),
    act_bits=int(act_bits),
    use_hadamard=int(use_hadamard),
    use_tbgr=int(use_tbgr),
    tbgr_outlier_factor=float(tbgr_outlier_factor),
    tbgr_outlier_sigma=float(tbgr_outlier_sigma),
    mixed_act_mode=mixed_act_mode,
    ctf_act_policy=ctf_act_policy,
    ctf_low_bits=int(ctf_low_bits),
    ctf_high_bits=int(ctf_high_bits),
    ctf_target_avg_bits=float(ctf_target_avg_bits),
    ctf_u_late_high_count=int(ctf_u_late_high_count),
)
sampler = HARTSampler(sampler_args)

manifest_file.parent.mkdir(parents=True, exist_ok=True)
with manifest_file.open("w", encoding="utf-8") as manifest:
    for index, prompt_file in enumerate(prompt_files, start=1):
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise SystemExit(f"Empty DPG prompt file: {prompt_file}")

        print(
            f"[{index}/{len(prompt_files)}] generating "
            f"{prompt_file.name}",
            flush=True,
        )
        sample_imgs, _generated_prompts = sampler.generate([prompt] * 4)
        sample_imgs = (
            sample_imgs.detach().clamp(0, 1).mul(255).round().to("cpu").to(torch.uint8)
        )
        images = [
            Image.fromarray(image.permute(1, 2, 0).numpy()).convert("RGB")
            for image in sample_imgs
        ]

        grid = Image.new("RGB", (resolution * 2, resolution * 2))
        for image_index, image in enumerate(images):
            if image.size != (resolution, resolution):
                image = image.resize(
                    (resolution, resolution), resampling.BICUBIC
                )
            grid.paste(
                image,
                ((image_index % 2) * resolution, (image_index // 2) * resolution),
            )

        image_name = f"{prompt_file.stem}.png"
        grid.save(image_dir / image_name)
        manifest.write(
            json.dumps(
                {
                    "prompt_index": index,
                    "source_file": prompt_file.name,
                    "prompt": prompt,
                    "image_file": image_name,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        manifest.flush()

print(f"Generated {len(prompt_files)} DPG 2x2 grids in {image_dir}")
PY
fi

EVAL_IMAGE_DIR="$IMAGE_DIR"
if [[ "$EVAL_ONLY" == "1" && -f "$MANIFEST_FILE" ]]; then
  STAGED_IMAGE_DIR="$RUN_DIR/.dpg_eval_images"
  "$PYTHON" - \
    "$IMAGE_DIR" \
    "$MANIFEST_FILE" \
    "$STAGED_IMAGE_DIR" <<'PY'
import json
import sys
from pathlib import Path


image_dir = Path(sys.argv[1])
manifest_file = Path(sys.argv[2])
staged_dir = Path(sys.argv[3])
staged_dir.mkdir(parents=True, exist_ok=True)

staged_count = 0
for line in manifest_file.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    record = json.loads(line)
    source_name = record.get("image_file")
    source_file = record.get("source_file")
    if not source_name or not source_file:
        continue

    source_path = image_dir / source_name
    target_name = f"{Path(source_file).stem}.png"
    target_path = staged_dir / target_name
    if not source_path.is_file():
        raise SystemExit(f"Manifest image is missing: {source_path}")
    if source_path.name == target_name and source_path.parent == image_dir:
        continue
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    target_path.symlink_to(source_path.resolve())
    staged_count += 1

if staged_count:
    print(f"Staged {staged_count} DPG images with original prompt names: {staged_dir}")
    print(staged_dir)
else:
    print("")
PY
  if [[ -n "$(find "$STAGED_IMAGE_DIR" -mindepth 1 -print -quit)" ]]; then
    EVAL_IMAGE_DIR="$STAGED_IMAGE_DIR"
  fi
fi

if [[ -n "${DPG_EVALUATOR:-}" ]]; then
  export DPG_RUN_DIR="$RUN_DIR"
  export DPG_IMAGE_DIR="$EVAL_IMAGE_DIR"
  export DPG_SOURCE_IMAGE_DIR="$IMAGE_DIR"
  export DPG_PROMPT_DIR
  export DPG_MANIFEST_FILE="$MANIFEST_FILE"
  export DPG_RESULT_FILE="$RESULT_FILE"
  export DPG_RESOLUTION
  export DPG_EVALUATOR_ROOT="${DPG_EVALUATOR_ROOT:-}"
  echo "Running DPG evaluator from DPG_EVALUATOR."
  bash -c "$DPG_EVALUATOR"
else
  accelerate_args=(
    --num_machines 1
    --num_processes "$DPG_PROCESSES"
    --mixed_precision fp16
    --dynamo_backend no
    --main_process_port "$DPG_PORT"
  )
  if (( DPG_PROCESSES > 1 )); then
    accelerate_args+=(--multi_gpu)
  fi
  echo "Running built-in DPG-Bench evaluator."
  "$ACCELERATE" launch "${accelerate_args[@]}" \
    "$ROOT_DIR/checkpoints/dpg_bench/compute_dpg_bench.py" \
    --image-root-path "$EVAL_IMAGE_DIR" \
    --resolution "$DPG_RESOLUTION" \
    --pic-num 4 \
    --vqa-model mplug \
    --csv "$DPG_CSV" \
    --res-path "$RESULT_FILE"
fi

echo "DPG output directory: $RUN_DIR"
