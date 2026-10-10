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
EXISTING_MJHQ_RUN_DIR="${EXISTING_MJHQ_RUN_DIR:-}"
EXISTING_MJHQ_IMAGE_DIR="${EXISTING_MJHQ_IMAGE_DIR:-}"
EXISTING_MJHQ_MANIFEST_FILE="${EXISTING_MJHQ_MANIFEST_FILE:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"

MJHQ_DATASET_DIR="${MJHQ_DATASET_DIR:-$ROOT_DIR/checkpoints/MJHQ-30K}"
MJHQ_METADATA="${MJHQ_METADATA:-$MJHQ_DATASET_DIR/meta_data.json}"
MJHQ_REFERENCE_DIR="${MJHQ_REFERENCE_DIR:-$MJHQ_DATASET_DIR/images}"
MJHQ_INCEPTION_CACHE="${MJHQ_INCEPTION_CACHE:-$MJHQ_DATASET_DIR/inception-2015-12-05.pt}"
MJHQ_REFERENCE_STATS_DIR="${MJHQ_REFERENCE_STATS_DIR:-$MJHQ_DATASET_DIR/cleanfid_stats}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints/hart-0.7b-1024px/llm}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-$ROOT_DIR/checkpoints/Qwen2-VL-1.5B-Instruct}"

# These values are part of the original HART MJHQ protocol.
RESOLUTION="${RESOLUTION:-1024}"
MJHQ_LIMIT="${MJHQ_LIMIT:-0}"
MJHQ_OFFSET="${MJHQ_OFFSET:-0}"
MJHQ_BATCH_SIZE="${MJHQ_BATCH_SIZE:-1}"
SEED=1
USE_EMA=1
MAX_TOKEN_LENGTH=300
USE_LLM_SYSTEM_PROMPT=1
CFG=4.5
MORE_SMOOTH=0
RUN_EVAL="${RUN_EVAL:-1}"
MJHQ_PER_CATEGORY="${MJHQ_PER_CATEGORY:-1}"
FID_BATCH_SIZE="${FID_BATCH_SIZE:-32}"
FID_NUM_WORKERS="${FID_NUM_WORKERS:-4}"
FID_DEVICE="${FID_DEVICE:-cuda}"
FID_MODE="${FID_MODE:-clean}"

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
  if [[ -z "$EXISTING_MJHQ_RUN_DIR" && -z "$EXISTING_MJHQ_IMAGE_DIR" ]]; then
    echo "EVAL_ONLY=1 requires EXISTING_MJHQ_RUN_DIR or EXISTING_MJHQ_IMAGE_DIR." >&2
    exit 2
  fi
  if [[ -n "$EXISTING_MJHQ_RUN_DIR" ]]; then
    RUN_DIR="$EXISTING_MJHQ_RUN_DIR"
    IMAGE_DIR="${EXISTING_MJHQ_IMAGE_DIR:-$RUN_DIR/images}"
  else
    IMAGE_DIR="$EXISTING_MJHQ_IMAGE_DIR"
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
MANIFEST_FILE="${EXISTING_MJHQ_MANIFEST_FILE:-$RUN_DIR/mjhq_manifest.jsonl}"
RESULT_FILE="${MJHQ_RESULT_FILE:-$RUN_DIR/mjhq_result.txt}"
RESULT_JSON="${MJHQ_RESULT_JSON:-$RUN_DIR/mjhq_result.json}"
GEN_STATS_DIR="${MJHQ_GEN_STATS_DIR:-$RUN_DIR/cleanfid_stats}"

if [[ ! -f "$MJHQ_METADATA" ]]; then
  echo "Missing MJHQ metadata file: $MJHQ_METADATA" >&2
  exit 2
fi

if [[ "$EVAL_ONLY" == "1" ]]; then
  if [[ ! -d "$IMAGE_DIR" ]]; then
    echo "Existing MJHQ image directory does not exist: $IMAGE_DIR" >&2
    exit 2
  fi
  echo "EVAL_ONLY=1; skip MJHQ model loading and image generation."
else
  mkdir -p "$IMAGE_DIR"

  "$PYTHON" - \
  "$MJHQ_METADATA" \
  "$IMAGE_DIR" \
  "$MANIFEST_FILE" \
  "$MJHQ_LIMIT" \
  "$MJHQ_OFFSET" \
  "$MJHQ_BATCH_SIZE" \
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
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from sample import HARTSampler


(
    metadata_path,
    image_dir,
    manifest_file,
    limit,
    offset,
    batch_size,
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

image_dir = Path(image_dir)
manifest_file = Path(manifest_file)
limit = int(limit)
offset = int(offset)
batch_size = int(batch_size)
if batch_size < 1:
    raise SystemExit("MJHQ_BATCH_SIZE must be >= 1")


def load_records():
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("MJHQ_METADATA must contain the official object format")

    records = []
    for source_id, item in payload.items():
        if not isinstance(item, dict) or "prompt" not in item or "category" not in item:
            raise SystemExit(f"Invalid MJHQ metadata item: {source_id}")
        records.append(
            {
                "source_id": str(source_id),
                "category": str(item["category"]),
                "prompt": str(item["prompt"]),
            }
        )
    return records


records = load_records()
records = records[offset:]
if limit > 0 and limit < len(records):
    by_category = defaultdict(list)
    for record in records:
        by_category[record.get("category", "")].append(record)

    selected = []
    index = 0
    categories = sorted(by_category)
    while len(selected) < limit:
        progressed = False
        for category in categories:
            bucket = by_category[category]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        index += 1
    records = selected
if len(records) < 1:
    raise SystemExit("No MJHQ prompts selected")

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
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_number, start in enumerate(range(0, len(records), batch_size), start=1):
        batch = records[start : start + batch_size]
        prompts = [record["prompt"] for record in batch]
        print(
            f"[{batch_number}/{total_batches}] generating "
            f"{len(batch)} MJHQ prompts",
            flush=True,
        )
        sample_imgs, _generated_prompts = sampler.generate(
            prompts, generation_seed=int(seed) + start
        )
        sample_imgs = (
            sample_imgs.detach().clamp(0, 1).mul(255).round().to("cpu").to(torch.uint8)
        )
        images = [
            Image.fromarray(image.permute(1, 2, 0).numpy()).convert("RGB")
            for image in sample_imgs
        ]

        for local_index, (record, image) in enumerate(zip(batch, images)):
            category = str(record.get("category", "")).strip()
            source_id = str(record["source_id"])
            relative_image_path = (
                Path(category) / f"{source_id}.png"
                if category
                else Path(f"{source_id}.png")
            )
            image_name = relative_image_path.as_posix()
            target_path = image_dir / relative_image_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(target_path)
            manifest.write(
                json.dumps(
                    {
                        **record,
                        "image_file": image_name,
                        "prompt_index": start + local_index + 1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        manifest.flush()

print(f"Generated {len(records)} MJHQ images in {image_dir}")
PY
fi

EVAL_IMAGE_DIR="$IMAGE_DIR"
if [[ "$EVAL_ONLY" == "1" && -f "$MANIFEST_FILE" ]]; then
  STAGED_IMAGE_DIR="$RUN_DIR/.mjhq_eval_images"
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
    source_id = record.get("source_id")
    category = str(record.get("category", "")).strip()
    if not source_name or not source_id:
        continue

    source_path = image_dir / source_name
    relative_target = (
        Path(category) / f"{source_id}.png"
        if category
        else Path(f"{source_id}.png")
    )
    target_path = staged_dir / relative_target
    if not source_path.is_file():
        raise SystemExit(f"Manifest image is missing: {source_path}")
    if source_path == image_dir / relative_target:
        continue
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    target_path.symlink_to(source_path.resolve())
    staged_count += 1

if staged_count:
    print(f"Staged {staged_count} MJHQ images in original category/id layout: {staged_dir}")
PY
  if [[ -n "$(find "$STAGED_IMAGE_DIR" -mindepth 1 -print -quit)" ]]; then
    EVAL_IMAGE_DIR="$STAGED_IMAGE_DIR"
  fi
fi

if [[ -n "${MJHQ_EVALUATOR:-}" ]]; then
  export MJHQ_RUN_DIR="$RUN_DIR"
  export MJHQ_IMAGE_DIR="$EVAL_IMAGE_DIR"
  export MJHQ_SOURCE_IMAGE_DIR="$IMAGE_DIR"
  export MJHQ_MANIFEST_FILE="$MANIFEST_FILE"
  export MJHQ_RESULT_FILE="$RESULT_FILE"
  export MJHQ_RESULT_JSON="$RESULT_JSON"
  export MJHQ_METADATA
  export MJHQ_REFERENCE_DIR
  echo "Running MJHQ evaluator from MJHQ_EVALUATOR."
  bash -c "$MJHQ_EVALUATOR"
elif [[ "$RUN_EVAL" == "1" ]]; then
  mkdir -p "$GEN_STATS_DIR"
  "$PYTHON" - \
    "$MJHQ_METADATA" \
    "$MJHQ_REFERENCE_DIR" \
    "$EVAL_IMAGE_DIR" \
    "$RESULT_FILE" \
    "$RESULT_JSON" \
    "$MANIFEST_FILE" \
    "$MJHQ_REFERENCE_STATS_DIR" \
    "$GEN_STATS_DIR" \
    "$RESOLUTION" \
    "$MJHQ_LIMIT" \
    "$MJHQ_OFFSET" \
    "$FID_BATCH_SIZE" \
    "$FID_NUM_WORKERS" \
    "$FID_DEVICE" \
    "$FID_MODE" \
    "$MJHQ_PER_CATEGORY" \
    "$MJHQ_INCEPTION_CACHE" <<'PY'
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from cleanfid import fid
from PIL import Image


(
    meta_path,
    ref_root,
    gen_root,
    result_file,
    result_json,
    manifest_file,
    ref_stats_dir,
    gen_stats_dir,
    resolution,
    limit,
    offset,
    batch_size,
    num_workers,
    device_name,
    fid_mode,
    per_category,
    inception_cache,
) = sys.argv[1:]

meta_path = Path(meta_path)
ref_root = Path(ref_root)
gen_root = Path(gen_root)
result_file = Path(result_file)
result_json = Path(result_json)
manifest_file = Path(manifest_file)
ref_stats_dir = Path(ref_stats_dir)
gen_stats_dir = Path(gen_stats_dir)
resolution = int(resolution)
limit = int(limit)
offset = int(offset)
batch_size = int(batch_size)
num_workers = int(num_workers)
per_category = bool(int(per_category))
inception_cache = Path(inception_cache)


def load_records(path, limit_value, offset_value):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("MJHQ metadata must use the official object format")

    records = [
        {
            "id": str(stem),
            "prompt": str(item["prompt"]),
            "category": str(item["category"]),
        }
        for stem, item in payload.items()
    ]
    records = records[offset_value:]
    if limit_value <= 0 or limit_value >= len(records):
        return records

    by_category = defaultdict(list)
    for record in records:
        by_category[record["category"]].append(record)

    selected = []
    index = 0
    categories = sorted(by_category)
    while len(selected) < limit_value:
        progressed = False
        for category in categories:
            bucket = by_category[category]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) == limit_value:
                    break
        if not progressed:
            break
        index += 1
    return selected


def write_manifest(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def ref_path(record):
    return ref_root / record["category"] / f'{record["id"]}.jpg'


def gen_path(record):
    return gen_root / record["category"] / f'{record["id"]}.png'


def validate_images(paths, label):
    missing = []
    bad = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                if image.size != (resolution, resolution):
                    bad.append((str(path), image.size))
        except Exception as exc:
            bad.append((str(path), repr(exc)))
    if missing or bad:
        print(f"{label} validation failed: missing={len(missing)}, bad={len(bad)}")
        if missing:
            print("First missing:", missing[:5])
        if bad:
            print("First bad:", bad[:5])
        raise SystemExit(1)


def files_fingerprint(paths):
    digest = hashlib.sha1()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def stats_from_files(paths, cache_path, label, feat_model, device):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = files_fingerprint(paths)
    if cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            cached_fingerprint = str(cached["fingerprint"])
            cached_count = int(cached["count"])
            if cached_fingerprint == fingerprint and cached_count == len(paths):
                print(f"Load cached {label} stats: {cache_path}")
                return cached["mu"], cached["sigma"]
        except (KeyError, OSError, ValueError):
            pass

    print(f"Compute {label} clean-fid features: {len(paths)} images")
    features = fid.get_files_features(
        [str(path) for path in paths],
        model=feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=fid_mode,
        description=f"{label}: ",
        verbose=True,
    )
    if features.shape[0] < 2:
        raise RuntimeError(f"{label} needs at least 2 images for covariance")
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    np.savez_compressed(
        cache_path,
        mu=mu,
        sigma=sigma,
        count=np.array(len(paths)),
        fingerprint=np.array(fingerprint),
    )
    print(f"Saved {label} stats: {cache_path}")
    return mu, sigma


tmp_inception = Path("/tmp/inception-2015-12-05.pt")
if inception_cache.exists() and not tmp_inception.exists():
    shutil.copy2(inception_cache, tmp_inception)

records = load_records(meta_path, limit, offset)
if len(records) < 2:
    raise RuntimeError("MJHQ evaluation needs at least 2 generated images")
write_manifest(records, manifest_file)

ref_files = [ref_path(record) for record in records]
gen_files = [gen_path(record) for record in records]
validate_images(ref_files, "Reference")
validate_images(gen_files, "Generated")

device = torch.device(device_name)
feat_model = fid.build_feature_extractor(fid_mode, device, use_dataparallel=False)

official_full = limit <= 0 and offset == 0 and len(records) == 30000
tag = "full" if official_full else f"limit{len(records)}_offset{offset}"
run_name = gen_root.parent.name

start_time = time.time()
mu_ref, sigma_ref = stats_from_files(
    ref_files,
    ref_stats_dir / f"mjhq30k_ref_{fid_mode}_{tag}.npz",
    f"reference overall {tag}",
    feat_model,
    device,
)
mu_gen, sigma_gen = stats_from_files(
    gen_files,
    gen_stats_dir / f"{run_name}_gen_{fid_mode}_{tag}.npz",
    f"generated overall {tag}",
    feat_model,
    device,
)
overall = float(fid.frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen))

category_scores = {}
skipped_categories = {}
if per_category:
    categories = sorted({record["category"] for record in records})
    for category in categories:
        category_records = [
            record for record in records if record["category"] == category
        ]
        if len(category_records) < 2:
            skipped_categories[category] = "fewer than 2 generated images"
            continue
        category_ref_files = [ref_path(record) for record in category_records]
        category_gen_files = [gen_path(record) for record in category_records]
        category_tag = (
            "full"
            if official_full
            else f"limit{len(category_records)}_offset{offset}"
        )
        mu_ref_c, sigma_ref_c = stats_from_files(
            category_ref_files,
            ref_stats_dir
            / f"mjhq30k_ref_{fid_mode}_{category}_{category_tag}.npz",
            f"reference {category} {category_tag}",
            feat_model,
            device,
        )
        mu_gen_c, sigma_gen_c = stats_from_files(
            category_gen_files,
            gen_stats_dir
            / f"{run_name}_gen_{fid_mode}_{category}_{category_tag}.npz",
            f"generated {category} {category_tag}",
            feat_model,
            device,
        )
        category_scores[category] = float(
            fid.frechet_distance(mu_ref_c, sigma_ref_c, mu_gen_c, sigma_gen_c)
        )

elapsed = time.time() - start_time
result = {
    "run_name": run_name,
    "generated_dir": str(gen_root),
    "reference_dir": str(ref_root),
    "meta_path": str(meta_path),
    "num_images": len(records),
    "limit": limit,
    "offset": offset,
    "official_full_mjhq30k": official_full,
    "fid_mode": fid_mode,
    "fid_device": device_name,
    "fid_batch_size": batch_size,
    "fid_num_workers": num_workers,
    "overall_fid": overall,
    "per_category_fid": category_scores,
    "skipped_categories": skipped_categories,
    "elapsed_seconds": elapsed,
}

result_file.parent.mkdir(parents=True, exist_ok=True)
result_json.parent.mkdir(parents=True, exist_ok=True)
result_json.write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

lines = [
    f"Run: {run_name}",
    f"Images: {len(records)}",
    f"Official full MJHQ-30K: {official_full}",
    f"FID mode: {fid_mode}",
    f"Overall FID: {overall:.6f}",
]
for category, score in sorted(category_scores.items()):
    lines.append(f"{category}: {score:.6f}")
if skipped_categories:
    lines.append(f"Skipped categories: {skipped_categories}")
lines.append(f"Elapsed seconds: {elapsed:.2f}")
lines.append(f"Generated dir: {gen_root}")
lines.append(f"Reference dir: {ref_root}")
result_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
PY
else
  echo "RUN_EVAL=0; MJHQ evaluation was not run."
fi

echo "MJHQ output directory: $RUN_DIR"
