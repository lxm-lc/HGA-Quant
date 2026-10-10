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
METADATA_FILE="${METADATA_FILE:-evaluation/gen_eval/prompts/evaluation_metadata.jsonl}"
DETECTOR_NAME="${DETECTOR_NAME:-mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco}"
DETECTOR_MODEL_DIR="${DETECTOR_MODEL_DIR:-evaluation/gen_eval/mask2former}"
DETECTOR_CONFIG="${DETECTOR_CONFIG:-evaluation/gen_eval/mask2former/${DETECTOR_NAME}.py}"
DETECTOR_CHECKPOINT="${DETECTOR_MODEL_DIR}/${DETECTOR_NAME}.pth"

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
ensure_file_exists "evaluation/gen_eval/object_names.txt" "Missing GenEval object class list"
ensure_file_exists "$DETECTOR_CONFIG" "Missing GenEval detector config"
ensure_file_exists "$DETECTOR_CHECKPOINT" "Missing GenEval detector checkpoint"

if [[ "$EVAL_ONLY" == "1" ]]; then
    [[ -n "$EXISTING_RUN_DIR" ]] || {
        echo "EVAL_ONLY=1 requires EXISTING_RUN_DIR" >&2
        exit 2
    }
    RUN_DIR="$EXISTING_RUN_DIR"
    ensure_directory_exists "$RUN_DIR" "EXISTING_RUN_DIR does not exist"
else
    ensure_file_exists "$METADATA_FILE" "Missing GenEval metadata"
    ensure_default_hf_file "$MODEL_PATH" "$DEFAULT_MODEL_PATH" \
        "FoundationVision/infinity" "infinity_2b_reg.pth"
    ensure_default_hf_file "$VAE_PATH" "$DEFAULT_VAE_PATH" \
        "FoundationVision/infinity" "infinity_vae_d32reg.pth"
    ensure_default_hf_directory "$TEXT_ENCODER_CKPT" "$DEFAULT_TEXT_ENCODER_CKPT" \
        "google/flan-t5-xl"
    RUN_DIR="$(create_run_dir "$OUTPUT_ROOT" "$QUANT_TAG")"
fi
mkdir -p "$RUN_DIR/results"

record_run_status() {
    update_run_record "$OUTPUT_ROOT" "$RUN_DIR" "geneval" "$1" \
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
        "metadata_file=$METADATA_FILE" \
        "detector_config=$DETECTOR_CONFIG" \
        "detector_checkpoint=$DETECTOR_CHECKPOINT" \
        "n_samples=4"
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

echo "[GenEval] output: $RUN_DIR"
record_run_status "in_progress"

if [[ "$EVAL_ONLY" == "1" ]]; then
    echo "[GenEval] evaluation-only mode; generation skipped"
else
    "${PYTHON_CMD[@]}" evaluation/gen_eval/infer4eval.py \
        "${COMMON_ARGS[@]}" \
        --outdir "$RUN_DIR" \
        --metadata_file "$METADATA_FILE" \
        --n_samples 4
fi

"${EVAL_PYTHON_CMD[@]}" evaluation/gen_eval/evaluate_images.py \
    "$RUN_DIR" \
    --outfile "$RUN_DIR/results/detections.jsonl" \
    --model-config "$DETECTOR_CONFIG" \
    --model-path "$DETECTOR_MODEL_DIR" \
    --options "model=$DETECTOR_NAME"

"${EVAL_PYTHON_CMD[@]}" evaluation/gen_eval/summary_scores.py \
    "$RUN_DIR/results/detections.jsonl" \
    | tee "$RUN_DIR/results/summary.txt"

record_run_status "completed"
echo "[GenEval] completed: $RUN_DIR"
