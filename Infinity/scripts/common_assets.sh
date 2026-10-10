#!/usr/bin/env bash

# Shared helpers for the Infinity evaluation scripts.

configure_hf_endpoint() {
    if [[ "${USE_HF_MIRROR:-0}" == "1" && -z "${HF_ENDPOINT:-}" ]]; then
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
}

download_hf_file() {
    local repo_id="$1"
    local filename="$2"
    local local_dir="$3"

    configure_hf_endpoint
    mkdir -p "$local_dir"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo_id" "$filename" --local-dir "$local_dir"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo_id" "$filename" --local-dir "$local_dir"
    else
        "${PYTHON_CMD[@]}" - "$repo_id" "$filename" "$local_dir" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id, filename, local_dir = sys.argv[1:]
hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
PY
    fi
}

download_hf_snapshot() {
    local repo_id="$1"
    local local_dir="$2"

    configure_hf_endpoint
    mkdir -p "$local_dir"
    if command -v hf >/dev/null 2>&1; then
        hf download "$repo_id" --local-dir "$local_dir"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "$repo_id" --local-dir "$local_dir"
    else
        "${PYTHON_CMD[@]}" - "$repo_id" "$local_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, local_dir = sys.argv[1:]
snapshot_download(repo_id=repo_id, local_dir=local_dir)
PY
    fi
}

ensure_default_hf_file() {
    local path="$1"
    local default_path="$2"
    local repo_id="$3"
    local filename="$4"

    if [[ -f "$path" ]]; then
        return
    fi
    if [[ "$path" != "$default_path" ]]; then
        echo "Missing custom resource: $path" >&2
        exit 2
    fi

    echo "[Infinity] missing $path; downloading $repo_id/$filename"
    download_hf_file "$repo_id" "$filename" "$(dirname "$path")"
    [[ -f "$path" ]] || {
        echo "Hugging Face download did not create $path" >&2
        exit 2
    }
}

ensure_default_hf_directory() {
    local path="$1"
    local default_path="$2"
    local repo_id="$3"

    if [[ -f "$path/config.json" ]]; then
        return
    fi
    if [[ "$path" != "$default_path" ]]; then
        echo "Missing custom resource directory: $path" >&2
        exit 2
    fi

    echo "[Infinity] missing $path; downloading $repo_id"
    download_hf_snapshot "$repo_id" "$path"
    [[ -f "$path/config.json" ]] || {
        echo "Hugging Face download did not create $path/config.json" >&2
        exit 2
    }
}

ensure_file_exists() {
    local path="$1"
    local message="${2:-Missing required file}"

    [[ -f "$path" ]] || {
        echo "$message: $path" >&2
        exit 2
    }
}

ensure_directory_exists() {
    local path="$1"
    local message="${2:-Missing required directory}"

    [[ -d "$path" ]] || {
        echo "$message: $path" >&2
        exit 2
    }
}

validate_eval_only() {
    local eval_only="$1"

    case "$eval_only" in
        0|1)
            ;;
        *)
            echo "EVAL_ONLY must be 0 or 1; got $eval_only" >&2
            exit 2
            ;;
    esac
}

validate_quant_config() {
    case "$MIXED_ACT_MODE" in
        default|coarse_to_fine)
            ;;
        *)
            echo "MIXED_ACT_MODE must be default or coarse_to_fine; got $MIXED_ACT_MODE" >&2
            exit 2
            ;;
    esac

    if [[ "$MIXED_ACT_MODE" == "coarse_to_fine" &&
          ( "$QUANTIZE_WEIGHT" != "1" || "$QUANTIZE_ACT" != "1" ) ]]; then
        echo "coarse_to_fine mode requires QUANTIZE_WEIGHT=1 and QUANTIZE_ACT=1." >&2
        exit 2
    fi
}

format_number() {
    printf '%s' "$1" | sed 's/\.0*$//; s/\./p/g'
}

compute_quant_tag() {
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
            ACT_TAG="$ACT_BITS"
            QUANT_TAG="W${WEIGHT_TAG}A${ACT_TAG}_default"
        fi
    else
        QUANT_TAG="W${WEIGHT_TAG}Afp_default"
    fi
}

create_run_dir() {
    local output_root="$1"
    local quant_tag="$2"
    local run_stem run_dir run_index

    mkdir -p "$output_root"
    run_stem="Infinity_HGA_${quant_tag}_$(date +%Y%m%d%H%M)"
    run_dir="$output_root/$run_stem"
    run_index=1
    while [[ -e "$run_dir" ]]; do
        run_dir="$output_root/${run_stem}_${run_index}"
        run_index=$((run_index + 1))
    done
    mkdir -p "$run_dir"
    printf '%s\n' "$run_dir"
}

update_run_record() {
    local output_root="$1"
    local run_dir="$2"
    local benchmark="$3"
    local status="$4"
    shift 4

    mkdir -p "$output_root" "$run_dir"
    "${PYTHON_CMD[@]}" - "$output_root/run_records.json" "$run_dir" "$benchmark" "$status" "$@" <<'PY'
import json
import os
import sys
from datetime import datetime

record_path, run_dir, benchmark, status, *pairs = sys.argv[1:]
now = datetime.now().isoformat(timespec="seconds")

config = {}
for item in pairs:
    if "=" not in item:
        continue
    key, value = item.split("=", 1)
    low = value.lower()
    if low in {"true", "false"}:
        parsed = low == "true"
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    config[key] = parsed

try:
    with open(record_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = []
except FileNotFoundError:
    records = []
except json.JSONDecodeError:
    records = []

matched = None
for record in records:
    if record.get("run_dir") == run_dir and record.get("benchmark") == benchmark:
        matched = record
        break

if matched is None:
    matched = {
        "benchmark": benchmark,
        "run_dir": run_dir,
        "started_at": now,
        "config": {},
    }
    records.append(matched)

matched["status"] = status
matched["updated_at"] = now
matched["config"].update(config)
if status == "completed":
    matched["finished_at"] = now

os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
with open(record_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
    f.write("\n")

with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as f:
    json.dump(matched, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}
