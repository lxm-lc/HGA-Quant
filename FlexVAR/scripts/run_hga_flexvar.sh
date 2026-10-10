#!/usr/bin/env bash
set -euo pipefail

# One generation task. Set environment variables before invoking this script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-${PYTHON_BIN}}"
NPROC_PER_NODE=1

DEPTH="${DEPTH:-24}"
case "${DEPTH}" in
    16)
        DEFAULT_VAR_CKPT="pretrained/FlexVARd16-epo179.pth"
        ;;
    20)
        DEFAULT_VAR_CKPT="pretrained/FlexVARd20-epo249.pth"
        ;;
    24)
        DEFAULT_VAR_CKPT="pretrained/FlexVARd24-epo349.pth"
        ;;
    *)
        DEFAULT_VAR_CKPT="pretrained/FlexVARd${DEPTH}.pth"
        ;;
esac
VAR_CKPT="${VAR_CKPT:-${PRETRAINED:-${DEFAULT_VAR_CKPT}}}"
VAE_CKPT="${VAE_CKPT:-pretrained/FlexVAE.pth}"
BATCH_SIZE=4
MAX_IMAGES=50000
SEED=0
CFG=4
TOP_K=900
MAX_PN=16
INFER_PATCH_NUMS=1_2_3_4_5_6_8_10_13_16

MIXED_ACT_MODE="${MIXED_ACT_MODE:-coarse_to_fine}"
QUANTIZE_WEIGHT="${QUANTIZE_WEIGHT:-1}"
QUANTIZE_ACT="${QUANTIZE_ACT:-1}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACT_BITS="${ACT_BITS:-6}"
USE_HADAMARD="${USE_HADAMARD:-1}"
USE_TBGR="${USE_TBGR:-1}"
TBGR_OUTLIER_FACTOR="${TBGR_OUTLIER_FACTOR:-8}"
TBGR_OUTLIER_SIGMA="${TBGR_OUTLIER_SIGMA:-3}"

case "${MIXED_ACT_MODE,,}" in
    default)
        MIXED_ACT_MODE=default
        ;;
    ctf|coarse_to_fine)
        MIXED_ACT_MODE=coarse_to_fine
        ;;
    *)
        echo "MIXED_ACT_MODE must be default or ctf/coarse_to_fine." >&2
        exit 2
        ;;
esac

CTF_ACT_POLICY="${CTF_ACT_POLICY:-early_high}"
CTF_LOW_BITS="${CTF_LOW_BITS:-4}"
CTF_HIGH_BITS="${CTF_HIGH_BITS:-8}"
CTF_TARGET_AVG_BITS="${CTF_TARGET_AVG_BITS:-6.0}"
CTF_U_LATE_HIGH_COUNT="${CTF_U_LATE_HIGH_COUNT:-1}"

EVAL_ONLY="${EVAL_ONLY:-0}"
EXISTING_SAMPLE_NPZ="${EXISTING_SAMPLE_NPZ:-}"
REFERENCE_NPZ="${REFERENCE_NPZ:-utils/evaluations/c2i/VIRTUAL_imagenet256_labeled.npz}"

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
        [[ "$QUANTIZE_WEIGHT" == "1" ]] || {
            echo "coarse_to_fine requires QUANTIZE_WEIGHT=1" >&2
            exit 2
        }
        ACT_TAG="$(format_number "$CTF_TARGET_AVG_BITS")"
        QUANT_TAG="W${WEIGHT_TAG}A${ACT_TAG}_ctf_low${CTF_LOW_BITS}high${CTF_HIGH_BITS}"
    else
        QUANT_TAG="W${WEIGHT_TAG}A${ACT_BITS}_default"
    fi
else
    [[ "$MIXED_ACT_MODE" != "coarse_to_fine" ]] || {
        echo "coarse_to_fine requires QUANTIZE_ACT=1" >&2
        exit 2
    }
    QUANT_TAG="W${WEIGHT_TAG}Afp_default"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"

case "${EVAL_ONLY}" in
    0)
        TIMESTAMP="$(date +%Y%m%d%H%M)"
        RUN_STEM="FlexVAR_HGA_${QUANT_TAG}_${TIMESTAMP}"
        TASK_DIR="${OUTPUT_ROOT}/${RUN_STEM}"
        RUN_INDEX=1
        while [[ -e "${TASK_DIR}" || -e "${TASK_DIR}.npz" ]]; do
            TASK_DIR="${OUTPUT_ROOT}/${RUN_STEM}_${RUN_INDEX}"
            RUN_INDEX=$((RUN_INDEX + 1))
        done
        SAMPLE_NPZ="${TASK_DIR}.npz"
        ;;
    1)
        [[ -n "${EXISTING_SAMPLE_NPZ}" ]] || {
            echo "EVAL_ONLY=1 requires EXISTING_SAMPLE_NPZ" >&2
            exit 2
        }
        SAMPLE_NPZ="${EXISTING_SAMPLE_NPZ}"
        [[ -f "${SAMPLE_NPZ}" ]] || {
            echo "EXISTING_SAMPLE_NPZ does not exist: ${SAMPLE_NPZ}" >&2
            exit 2
        }
        TASK_DIR="${SAMPLE_NPZ%.npz}"
        ;;
    *)
        echo "EVAL_ONLY must be 0 or 1; got ${EVAL_ONLY}" >&2
        exit 2
        ;;
esac

if [[ -z "${TORCHRUN_BIN}" ]]; then
    TORCHRUN_CMD=("${PYTHON_BIN}" -m torch.distributed.run)
else
    read -r -a TORCHRUN_CMD <<< "${TORCHRUN_BIN}"
fi

if [[ -n "${EVAL_SCRIPT:-}" && ! -f "${EVAL_SCRIPT}" ]]; then
    echo "EVAL_SCRIPT does not exist: ${EVAL_SCRIPT}" >&2
    exit 2
fi
if [[ -n "${REFERENCE_NPZ:-}" && ! -f "${REFERENCE_NPZ}" ]]; then
    echo "REFERENCE_NPZ does not exist: ${REFERENCE_NPZ}" >&2
    exit 2
fi
if [[ "${EVAL_ONLY}" == "0" ]]; then
    if [[ ! -f "${VAR_CKPT}" ]]; then
        echo "VAR_CKPT/PRETRAINED does not exist: ${VAR_CKPT}" >&2
        exit 2
    fi
    if [[ ! -f "${VAE_CKPT}" ]]; then
        echo "VAE_CKPT does not exist: ${VAE_CKPT}" >&2
        exit 2
    fi
fi

export TOKENIZERS_PARALLELISM=false

echo "Repository: ${REPO_ROOT}"
echo "Task directory: ${TASK_DIR}"
echo "Sample archive: ${SAMPLE_NPZ}"
echo "mixed_act_mode=${MIXED_ACT_MODE}"
if [[ "${EVAL_ONLY}" == "1" ]]; then
    echo "Evaluation-only mode: generation skipped"
else
    echo "Generating ${MAX_IMAGES} images with seed=${SEED}"
    "${TORCHRUN_CMD[@]}" \
        --nnodes=1 \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --node_rank=0 \
        eval_c2i.py \
        --batch_size "${BATCH_SIZE}" \
        --cfg "${CFG}" \
        --top_k "${TOP_K}" \
        --maxpn "${MAX_PN}" \
        --infer_patch_nums "${INFER_PATCH_NUMS//-/_}" \
        --var_ckpt "${VAR_CKPT}" \
        --vae_ckpt "${VAE_CKPT}" \
        --depth "${DEPTH}" \
        --max_images "${MAX_IMAGES}" \
        --seed "${SEED}" \
        --output_root "${OUTPUT_ROOT}" \
        --exp_name "$(basename -- "${TASK_DIR}")" \
        --quantize_weight "${QUANTIZE_WEIGHT}" \
        --quantize_act "${QUANTIZE_ACT}" \
        --weight_bits "${WEIGHT_BITS}" \
        --act_bits "${ACT_BITS}" \
        --use_hadamard "${USE_HADAMARD}" \
        --use_tbgr "${USE_TBGR}" \
        --tbgr_outlier_factor "${TBGR_OUTLIER_FACTOR}" \
        --tbgr_outlier_sigma "${TBGR_OUTLIER_SIGMA}" \
        --mixed_act_mode "${MIXED_ACT_MODE}" \
        --ctf_act_policy "${CTF_ACT_POLICY}" \
        --ctf_low_bits "${CTF_LOW_BITS}" \
        --ctf_high_bits "${CTF_HIGH_BITS}" \
        --ctf_target_avg_bits "${CTF_TARGET_AVG_BITS}" \
        --ctf_u_late_high_count "${CTF_U_LATE_HIGH_COUNT}"
fi

if [[ ! -f "${SAMPLE_NPZ}" ]]; then
    echo "Evaluation input missing: generated sample archive was not found at ${SAMPLE_NPZ}." >&2
    exit 1
fi

EVAL_ARGS=(--ref_batch "${REFERENCE_NPZ}" --sample_batch "${SAMPLE_NPZ}")

if [[ -n "${EVAL_SCRIPT:-}" ]]; then
    echo "Running external evaluator: ${EVAL_SCRIPT}"
    if [[ "${EVAL_SCRIPT}" == *.py ]]; then
        "${EVAL_PYTHON_BIN}" "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"
    elif [[ -x "${EVAL_SCRIPT}" ]]; then
        "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"
    else
        bash "${EVAL_SCRIPT}" "${EVAL_ARGS[@]}"
    fi
else
    echo "Running repository evaluator: utils/evaluations/c2i/evaluator.py"
    "${EVAL_PYTHON_BIN}" utils/evaluations/c2i/evaluator.py "${EVAL_ARGS[@]}"
fi
