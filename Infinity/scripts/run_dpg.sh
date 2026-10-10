#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$PYTHON_BIN}"
PYTHON_CMD=("$PYTHON_BIN")
EVAL_PYTHON_CMD=("$EVAL_PYTHON_BIN")
if [[ -n "${PYTHON_CONDA_ENV:-}" ]]; then
    PYTHON_CMD=(conda run --no-capture-output -n "$PYTHON_CONDA_ENV" python)
fi
if [[ -n "${EVAL_CONDA_ENV:-}" ]]; then
    EVAL_PYTHON_CMD=(conda run --no-capture-output -n "$EVAL_CONDA_ENV" python)
fi
source "$SCRIPT_DIR/common_assets.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
EVAL_ONLY="${EVAL_ONLY:-0}"
EXISTING_RUN_DIR="${EXISTING_RUN_DIR:-}"
PROMPT_DIR="${PROMPT_DIR:-evaluation/dpg_bench/prompts}"
DPG_CSV="${DPG_CSV:-evaluation/dpg_bench/dpg_bench.csv}"
DPG_VQA_MODEL="${DPG_VQA_MODEL:-mplug}"
SAVE_INDIVIDUAL="${SAVE_INDIVIDUAL:-0}"

DEFAULT_MODEL_PATH="weights/infinity_2b_reg.pth"
DEFAULT_VAE_PATH="weights/infinity_vae_d32reg.pth"
DEFAULT_TEXT_ENCODER_CKPT="weights/flan-t5-xl"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
VAE_PATH="${VAE_PATH:-$DEFAULT_VAE_PATH}"
TEXT_ENCODER_CKPT="${TEXT_ENCODER_CKPT:-$DEFAULT_TEXT_ENCODER_CKPT}"

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

validate_eval_only "$EVAL_ONLY"
validate_quant_config
compute_quant_tag
ensure_file_exists "$DPG_CSV" "Missing DPG-Bench csv"

if [[ "$EVAL_ONLY" == "1" ]]; then
    [[ -n "$EXISTING_RUN_DIR" ]] || {
        echo "EVAL_ONLY=1 requires EXISTING_RUN_DIR" >&2
        exit 2
    }
    RUN_DIR="$EXISTING_RUN_DIR"
    ensure_directory_exists "$RUN_DIR" "EXISTING_RUN_DIR does not exist"
else
    ensure_directory_exists "$PROMPT_DIR" "Missing DPG-Bench prompt directory"
    ensure_default_hf_file "$MODEL_PATH" "$DEFAULT_MODEL_PATH" \
        "FoundationVision/infinity" "infinity_2b_reg.pth"
    ensure_default_hf_file "$VAE_PATH" "$DEFAULT_VAE_PATH" \
        "FoundationVision/infinity" "infinity_vae_d32reg.pth"
    ensure_default_hf_directory "$TEXT_ENCODER_CKPT" "$DEFAULT_TEXT_ENCODER_CKPT" \
        "google/flan-t5-xl"
    RUN_DIR="$(create_run_dir "$OUTPUT_ROOT" "$QUANT_TAG")"
fi

record_run_status() {
    update_run_record "$OUTPUT_ROOT" "$RUN_DIR" "dpg" "$1" \
        "quant_tag=$QUANT_TAG" \
        "eval_only=$EVAL_ONLY" \
        "model_path=$MODEL_PATH" \
        "vae_path=$VAE_PATH" \
        "text_encoder_ckpt=$TEXT_ENCODER_CKPT" \
        "quantize_weight=$QUANTIZE_WEIGHT" \
        "quantize_act=$QUANTIZE_ACT" \
        "weight_bits=$WEIGHT_BITS" \
        "act_bits=$ACT_BITS" \
        "use_hadamard=$USE_HADAMARD" \
        "use_tbgr=$USE_TBGR" \
        "tbgr_outlier_factor=$TBGR_OUTLIER_FACTOR" \
        "tbgr_outlier_sigma=$TBGR_OUTLIER_SIGMA" \
        "mixed_act_mode=$MIXED_ACT_MODE" \
        "ctf_act_policy=$CTF_ACT_POLICY" \
        "ctf_low_bits=$CTF_LOW_BITS" \
        "ctf_high_bits=$CTF_HIGH_BITS" \
        "ctf_target_avg_bits=$CTF_TARGET_AVG_BITS" \
        "ctf_u_late_high_count=$CTF_U_LATE_HIGH_COUNT" \
        "prompt_dir=$PROMPT_DIR" \
        "dpg_csv=$DPG_CSV" \
        "n_samples=4" \
        "save_individual=$SAVE_INDIVIDUAL"
}

COMMON_ARGS=(
    --cfg 4
    --tau 1
    --pn 1M
    --model_path "$MODEL_PATH"
    --vae_type 32
    --vae_path "$VAE_PATH"
    --add_lvl_embeding_only_first_block 1
    --use_bit_label 1
    --model_type infinity_2b
    --rope2d_each_sa_layer 1
    --rope2d_normalized_by_hw 2
    --use_scale_schedule_embedding 0
    --checkpoint_type torch
    --text_encoder_ckpt "$TEXT_ENCODER_CKPT"
    --text_channels 2048
    --apply_spatial_patchify 0
    --cfg_insertion_layer 0
    --seed 0
    --quantize_weight "$QUANTIZE_WEIGHT"
    --quantize_act "$QUANTIZE_ACT"
    --weight_bits "$WEIGHT_BITS"
    --act_bits "$ACT_BITS"
    --use_hadamard "$USE_HADAMARD"
    --use_tbgr "$USE_TBGR"
    --tbgr_outlier_factor "$TBGR_OUTLIER_FACTOR"
    --tbgr_outlier_sigma "$TBGR_OUTLIER_SIGMA"
    --mixed_act_mode "$MIXED_ACT_MODE"
    --ctf_act_policy "$CTF_ACT_POLICY"
    --ctf_low_bits "$CTF_LOW_BITS"
    --ctf_high_bits "$CTF_HIGH_BITS"
    --ctf_target_avg_bits "$CTF_TARGET_AVG_BITS"
    --ctf_u_late_high_count "$CTF_U_LATE_HIGH_COUNT"
)

echo "[DPG] output: $RUN_DIR"
record_run_status "in_progress"

if [[ "$EVAL_ONLY" == "1" ]]; then
    echo "[DPG] evaluation-only mode; generation skipped"
else
    "${PYTHON_CMD[@]}" evaluation/dpg_bench/infer4eval.py \
        "${COMMON_ARGS[@]}" \
        --outdir "$RUN_DIR" \
        --prompt_dir "$PROMPT_DIR" \
        --grid_resolution 1024 \
        --n_samples 4 \
        --save_individual "$SAVE_INDIVIDUAL"
fi

ensure_directory_exists "$RUN_DIR/images" "Missing DPG image directory"
"${EVAL_PYTHON_CMD[@]}" evaluation/dpg_bench/compute_dpg_bench.py \
    --image-root-path "$RUN_DIR/images" \
    --resolution 1024 \
    --pic-num 4 \
    --vqa-model "$DPG_VQA_MODEL" \
    --csv "$DPG_CSV" \
    --res-path "$RUN_DIR/dpg_result.txt" \
    | tee "$RUN_DIR/dpg_eval.log"

record_run_status "completed"
echo "[DPG] completed: $RUN_DIR"
