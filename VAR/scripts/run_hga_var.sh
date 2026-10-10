#!/usr/bin/env bash

if [[ -z "${BASH_VERSION:-}" ]]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

die() {
    echo "[HGA-VAR] error: $*" >&2
    exit 1
}

sanitize_task_name() {
    local value="$1"
    value="$(printf '%s' "${value}" | sed 's/[^A-Za-z0-9_]/_/g; s/^_*//; s/_*$//')"
    [[ -n "${value}" ]] || die "TASK_NAME must contain at least one alphanumeric or underscore character"
    printf '%s' "${value}"
}

rename_images_with_underscore_prefix() {
    local out_dir="$1"
    local prefix="$2"
    local path base stem target renamed=0
    shopt -s nullglob
    for path in "${out_dir}"/[0-9][0-9][0-9][0-9][0-9].png; do
        base="$(basename "${path}")"
        stem="${base%.png}"
        target="${out_dir}/${prefix}_${stem}.png"
        [[ ! -e "${target}" ]] || die "cannot rename ${base}; target already exists: ${target}"
        mv -- "${path}" "${target}"
        renamed=$((renamed + 1))
    done
    shopt -u nullglob
    echo "[HGA-VAR] underscore_image_names=${renamed} prefix=${prefix}"
}

build_interleaved_sample_npz() {
    local sample_dir="$1"
    local npz_path="$2"
    local expected_count="$3"
    local resolution="$4"
    local num_classes=1000
    local samples_per_class=50

    echo "[NPZ] packing official ImageNet256 order: ${sample_dir} -> ${npz_path}"
    "${PYTHON_CMD[@]}" - \
        "${sample_dir}" \
        "${npz_path}" \
        "${expected_count}" \
        "${resolution}" \
        "${num_classes}" \
        "${samples_per_class}" <<'PY'
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


sample_dir = Path(sys.argv[1])
npz_path = Path(sys.argv[2])
expected_count = int(sys.argv[3])
resolution = int(sys.argv[4])
num_classes = int(sys.argv[5])
samples_per_class = int(sys.argv[6])

if expected_count != num_classes * samples_per_class:
    raise ValueError(
        f'official ImageNet256 ordering expects {num_classes * samples_per_class} '
        f'samples, got {expected_count}'
    )

pngs = sorted(
    [path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() == '.png'],
    key=lambda path: path.name,
)
if len(pngs) != expected_count:
    raise ValueError(f'{sample_dir} has {len(pngs)} PNG files, expected {expected_count}')

npz_path.parent.mkdir(parents=True, exist_ok=True)
tmp_npy = npz_path.with_name(npz_path.name + '.arr_0.npy.tmp')
tmp_npz = npz_path.with_name(npz_path.name + '.tmp')
for path in (tmp_npy, tmp_npz):
    if path.exists():
        path.unlink()

arr = np.lib.format.open_memmap(
    tmp_npy,
    mode='w+',
    dtype=np.uint8,
    shape=(expected_count, resolution, resolution, 3),
)

# Generation is class-major (50 images for class 0, then 50 for class 1, ...).
# Official IS evaluation uses repeat-major ordering so each split contains all classes.
for repeat_id in range(samples_per_class):
    for class_id in range(num_classes):
        source_index = class_id * samples_per_class + repeat_id
        destination_index = repeat_id * num_classes + class_id
        with Image.open(pngs[source_index]) as image:
            image = image.convert('RGB')
            if image.size != (resolution, resolution):
                raise ValueError(
                    f'{pngs[source_index]} has size {image.size}, '
                    f'expected {(resolution, resolution)}'
                )
            arr[destination_index] = np.asarray(image, dtype=np.uint8)

arr.flush()
del arr

with zipfile.ZipFile(tmp_npz, mode='w', compression=zipfile.ZIP_STORED) as zip_file:
    zip_file.write(tmp_npy, arcname='arr_0.npy')
tmp_npz.replace(npz_path)
tmp_npy.unlink()
print(f'[NPZ] saved {npz_path} with repeat-major ImageNet256 ordering')
PY
}

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
DEPTH="${DEPTH:-30}"
VAR_CKPT="${VAR_CKPT:-}"
VAE_CKPT="${VAE_CKPT:-pretrained_models/vae_ch160v4096z32.pth}"
NUM_IMAGES=50000
BATCH_SIZE=20
SEED=0
QUANT_SEED=0
QUANTIZE_WEIGHT="${QUANTIZE_WEIGHT:-1}"
QUANTIZE_ACT="${QUANTIZE_ACT:-1}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACT_BITS="${ACT_BITS:-6}"
MIXED_ACT_MODE="${MIXED_ACT_MODE:-coarse_to_fine}"
CTF_ACT_POLICY="${CTF_ACT_POLICY:-early_high}"
CTF_LOW_BITS="${CTF_LOW_BITS:-4}"
CTF_HIGH_BITS="${CTF_HIGH_BITS:-8}"
CTF_TARGET_AVG_BITS="${CTF_TARGET_AVG_BITS:-6.0}"
CTF_U_LATE_HIGH_COUNT="${CTF_U_LATE_HIGH_COUNT:-1}"
USE_HADAMARD="${USE_HADAMARD:-1}"
USE_TBGR="${USE_TBGR:-1}"
TBGR_OUTLIER_FACTOR="${TBGR_OUTLIER_FACTOR:-8.0}"
TBGR_OUTLIER_SIGMA="${TBGR_OUTLIER_SIGMA:-3.0}"
IMAGE_NAME_PREFIX="sample"
DEFAULT_REFERENCE_NPZ="evaluation/VIRTUAL_imagenet256_labeled.npz"
REFERENCE_NPZ="${REFERENCE_NPZ:-}"
INCEPTION_PB="${INCEPTION_PB:-}"
EVAL_RESULT_FILE="${EVAL_RESULT_FILE:-}"
EXPECTED_COUNT="${NUM_IMAGES}"
EVAL_RESOLUTION=256
EVAL_ONLY="${EVAL_ONLY:-0}"
EXISTING_SAMPLE_DIR="${EXISTING_SAMPLE_DIR:-}"

if [[ -z "${REFERENCE_NPZ}" && -f "${DEFAULT_REFERENCE_NPZ}" ]]; then
    REFERENCE_NPZ="${DEFAULT_REFERENCE_NPZ}"
fi

case "${EVAL_ONLY}" in
    0|1)
        ;;
    *)
        die "EVAL_ONLY must be 0 or 1; got ${EVAL_ONLY}"
        ;;
esac

case "${MIXED_ACT_MODE,,}" in
    default)
        MIXED_ACT_MODE="default"
        ;;
    ctf|coarse_to_fine)
        MIXED_ACT_MODE="coarse_to_fine"
        ;;
    *)
        die "MIXED_ACT_MODE must be default or ctf/coarse_to_fine; got ${MIXED_ACT_MODE}"
        ;;
esac

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
        [[ "$QUANTIZE_WEIGHT" == "1" ]] || die "coarse_to_fine requires QUANTIZE_WEIGHT=1"
        ACT_TAG="$(format_number "$CTF_TARGET_AVG_BITS")"
        QUANT_TAG="W${WEIGHT_TAG}A${ACT_TAG}_ctf_low${CTF_LOW_BITS}high${CTF_HIGH_BITS}"
    else
        QUANT_TAG="W${WEIGHT_TAG}A${ACT_BITS}_default"
    fi
else
    [[ "$MIXED_ACT_MODE" != "coarse_to_fine" ]] ||
        die "coarse_to_fine requires QUANTIZE_ACT=1"
    QUANT_TAG="W${WEIGHT_TAG}Afp_default"
fi

IMAGE_NAME_PREFIX="$(sanitize_task_name "${IMAGE_NAME_PREFIX}")"
if [[ "${EVAL_ONLY}" == "1" ]]; then
    [[ -n "${EXISTING_SAMPLE_DIR}" ]] || die "EVAL_ONLY=1 requires EXISTING_SAMPLE_DIR"
    RUN_DIR="${EXISTING_SAMPLE_DIR}"
    [[ -d "${RUN_DIR}" ]] || die "EXISTING_SAMPLE_DIR does not exist: ${RUN_DIR}"
else
    TIMESTAMP="$(date '+%Y%m%d%H%M')"
    RUN_STEM="VAR_HGA_${QUANT_TAG}_${TIMESTAMP}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_STEM}"
    RUN_INDEX=1
    while [[ -e "${RUN_DIR}" ]]; do
        RUN_DIR="${OUTPUT_ROOT}/${RUN_STEM}_${RUN_INDEX}"
        RUN_INDEX=$((RUN_INDEX + 1))
    done
    mkdir -p "${RUN_DIR}"
fi
TASK_NAME="$(basename -- "${RUN_DIR}")"

PYTHON_CMD=()
read -r -a PYTHON_CMD <<< "${PYTHON_BIN}"
[[ "${#PYTHON_CMD[@]}" -gt 0 ]] || die "PYTHON_BIN is empty"

CMD=(
    "${PYTHON_CMD[@]}"
    sample_var.py
    --out_dir "${RUN_DIR}"
    --num_images "${NUM_IMAGES}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --depth "${DEPTH}"
    --device cuda
    --fp16 1
    --fuse 1
    --pn 1_2_3_4_5_6_8_10_13_16
    --num_classes 1000
    --label_mode balanced
    --samples_per_class 50
    --class_start 0
    --class_end 999
    --cfg 1.5
    --top_k 900
    --top_p 0.96
    --more_smooth 0
    --quantize_weight "${QUANTIZE_WEIGHT}"
    --quantize_act "${QUANTIZE_ACT}"
    --quant_seed "${QUANT_SEED}"
    --weight_bits "${WEIGHT_BITS}"
    --act_bits "${ACT_BITS}"
    --mixed_act_mode "${MIXED_ACT_MODE}"
    --ctf_act_policy "${CTF_ACT_POLICY}"
    --ctf_low_bits "${CTF_LOW_BITS}"
    --ctf_high_bits "${CTF_HIGH_BITS}"
    --ctf_target_avg_bits "${CTF_TARGET_AVG_BITS}"
    --ctf_u_late_high_count "${CTF_U_LATE_HIGH_COUNT}"
    --use_hadamard "${USE_HADAMARD}"
    --use_tbgr "${USE_TBGR}"
    --tbgr_outlier_factor "${TBGR_OUTLIER_FACTOR}"
    --tbgr_outlier_sigma "${TBGR_OUTLIER_SIGMA}"
)

if [[ -n "${VAR_CKPT}" ]]; then
    CMD+=(--var_ckpt "${VAR_CKPT}")
fi
CMD+=(--vae_ckpt "${VAE_CKPT}")

echo "[HGA-VAR] repo=${REPO_ROOT}"
echo "[HGA-VAR] task=${TASK_NAME}"
if [[ "${EVAL_ONLY}" == "1" ]]; then
    echo "[HGA-VAR] input=${RUN_DIR}"
else
    echo "[HGA-VAR] output=${RUN_DIR}"
fi
echo "[HGA-VAR] mode=${MIXED_ACT_MODE}"
if [[ "${EVAL_ONLY}" == "1" ]]; then
    echo "[HGA-VAR] evaluation_only=1; generation skipped"
else
    printf '[HGA-VAR] command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    "${CMD[@]}"
    rename_images_with_underscore_prefix "${RUN_DIR}" "${IMAGE_NAME_PREFIX}"
fi

if [[ -n "${EVAL_SCRIPT:-}" || -n "${REFERENCE_NPZ}" ]]; then
    EVAL_CMD=()
    if [[ -z "${EVAL_SCRIPT:-}" ]]; then
        [[ -f "${REFERENCE_NPZ}" ]] || die "REFERENCE_NPZ does not exist: ${REFERENCE_NPZ}"
        EVAL_RESULT_FILE="${EVAL_RESULT_FILE:-${RUN_DIR}/var_metrics.txt}"
        EVAL_SAMPLE_NPZ="${RUN_DIR}.npz"
        build_interleaved_sample_npz \
            "${RUN_DIR}" \
            "${EVAL_SAMPLE_NPZ}" \
            "${EXPECTED_COUNT}" \
            "${EVAL_RESOLUTION}"
        EVAL_CMD=(
            "${PYTHON_CMD[@]}"
            eval_var_samples.py
            --sample_dir "${RUN_DIR}"
            --result_file "${EVAL_RESULT_FILE}"
            --reference_npz "${REFERENCE_NPZ}"
            --sample_npz "${EVAL_SAMPLE_NPZ}"
            --save_npz 1
            --expected_count "${EXPECTED_COUNT}"
            --resolution "${EVAL_RESOLUTION}"
            --require_official_imagenet256 1
            --metric_mode official_imagenet256
        )
        if [[ -n "${INCEPTION_PB}" ]]; then
            EVAL_CMD+=(--inception_pb "${INCEPTION_PB}")
        fi
    elif [[ -f "${EVAL_SCRIPT}" ]]; then
        case "${EVAL_SCRIPT##*.}" in
            py)
                EVAL_CMD=("${PYTHON_CMD[@]}" "${EVAL_SCRIPT}")
                ;;
            *)
                if [[ -x "${EVAL_SCRIPT}" ]]; then
                    EVAL_CMD=("${EVAL_SCRIPT}")
                else
                    EVAL_CMD=(bash "${EVAL_SCRIPT}")
                fi
                ;;
        esac
    else
        read -r -a EVAL_CMD <<< "${EVAL_SCRIPT}"
    fi
    [[ "${#EVAL_CMD[@]}" -gt 0 ]] || die "EVAL_SCRIPT is empty"

    export SAMPLE_DIR="${RUN_DIR}"
    export HGA_VAR_OUTPUT_DIR="${RUN_DIR}"
    EVAL_POSITIONAL_ARGS=()
    if [[ -n "${EVAL_SCRIPT:-}" ]]; then
        EVAL_POSITIONAL_ARGS=("${RUN_DIR}")
    fi
    echo "[Eval] running ${EVAL_SCRIPT:-eval_var_samples.py} with sample directory ${RUN_DIR}"
    printf '[Eval] command:'
    printf ' %q' "${EVAL_CMD[@]}" "${EVAL_POSITIONAL_ARGS[@]}"
    printf '\n'
    "${EVAL_CMD[@]}" "${EVAL_POSITIONAL_ARGS[@]}"
else
    if [[ "${EVAL_ONLY}" == "1" ]]; then
        die "EVAL_ONLY=1 requires REFERENCE_NPZ or EVAL_SCRIPT"
    fi
    echo "[Eval] skipped: neither EVAL_SCRIPT nor REFERENCE_NPZ is set"
fi

echo "[HGA-VAR] done: ${RUN_DIR}"
