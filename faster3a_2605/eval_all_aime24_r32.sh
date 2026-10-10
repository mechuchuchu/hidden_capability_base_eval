#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_LIST="${DOWNLOAD_LIST:-$SCRIPT_DIR/download_list.txt}"
MODEL_DIR="${MODEL_DIR:-/dev/shm}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/aime24_runs}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AIME_GPUS="${AIME_GPUS:-0}"
ROLLOUT=32
SEED=42
TOTAL_TASKS=30
TOTAL_GENERATIONS=$((TOTAL_TASKS * ROLLOUT))

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

for required_command in hf df tee; do
    command -v "$required_command" >/dev/null 2>&1 || die "Required command not found: $required_command"
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
[[ -r "$DOWNLOAD_LIST" ]] || die "Cannot read download list: $DOWNLOAD_LIST"
[[ -f "$SCRIPT_DIR/eval_aime24.py" ]] || die "Missing evaluator: $SCRIPT_DIR/eval_aime24.py"

mkdir -p -- "$MODEL_DIR" "$OUT_ROOT" "$LOG_ROOT"

declare -a MODEL_URIS=()
declare -A SEEN_URIS=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    if [[ "$line" =~ ^[[:space:]]*hf[[:space:]]+download[[:space:]]+(hf://[^[:space:]]+)[[:space:]]*$ ]]; then
        uri="${BASH_REMATCH[1]}"
    else
        die "Unsupported line in $DOWNLOAD_LIST: $line"
    fi

    if [[ -z "${SEEN_URIS[$uri]+present}" ]]; then
        SEEN_URIS["$uri"]=1
        MODEL_URIS+=("$uri")
    fi
done < "$DOWNLOAD_LIST"

(( ${#MODEL_URIS[@]} > 0 )) || die "No model downloads found in $DOWNLOAD_LIST"

is_complete_run() {
    local summary_path="$1"
    local generations_path="$2"
    local expected_model="$3"

    [[ -s "$summary_path" && -s "$generations_path" ]] || return 1
    "$PYTHON_BIN" - "$summary_path" "$generations_path" "$expected_model" "$TOTAL_TASKS" "$ROLLOUT" "$TOTAL_GENERATIONS" <<'PY'
import json
import pathlib
import sys

summary_path, generations_path, expected_model = sys.argv[1:4]
expected_tasks, expected_rollout, expected_generations = map(int, sys.argv[4:7])
try:
    summary = json.loads(pathlib.Path(summary_path).read_text(encoding="utf-8"))
    config = summary.get("config", {})
    if summary.get("num_tasks") != expected_tasks:
        raise SystemExit(1)
    if summary.get("rollout") != expected_rollout:
        raise SystemExit(1)
    if summary.get("total_generations") != expected_generations:
        raise SystemExit(1)
    if pathlib.Path(config.get("model", "")).name != expected_model:
        raise SystemExit(1)
    required_ks = {f"pass@{k}" for k in (1, 2, 4, 8, 16, 32)}
    if not required_ks.issubset(summary.get("pass_at_k", {})):
        raise SystemExit(1)
    with pathlib.Path(generations_path).open(encoding="utf-8") as f:
        rows = sum(1 for line in f if line.strip())
    if rows != expected_generations:
        raise SystemExit(1)
except (OSError, ValueError, TypeError, json.JSONDecodeError):
    raise SystemExit(1)
PY
}

model_count=${#MODEL_URIS[@]}
printf 'AIME24 full sweep: %d unique checkpoints, rollout=%d, total generations/model=%d\n' \
    "$model_count" "$ROLLOUT" "$TOTAL_GENERATIONS"
printf 'Downloads: %s\nOutputs: %s\nLogs: %s\n' "$MODEL_DIR" "$OUT_ROOT" "$LOG_ROOT"

index=0
for uri in "${MODEL_URIS[@]}"; do
    ((index += 1))
    filename="${uri##*/}"
    [[ "$filename" == *.pth ]] || die "Expected a .pth checkpoint, got: $filename"
    model_stem="${filename%.pth}"

    case "$filename" in
        *-1.5b-*) bsz=960; reserve_gib=4 ;;
        *-2.9b-*) bsz=960; reserve_gib=7 ;;
        *-7.2b-*) bsz=900; reserve_gib=16 ;;
        *-13.3b-*) bsz=240; reserve_gib=29 ;;
        *) die "No measured batch size for checkpoint: $filename" ;;
    esac

    model_path="$MODEL_DIR/$filename"
    out_dir="$OUT_ROOT/${model_stem}-r${ROLLOUT}-seed${SEED}"
    summary_path="$out_dir/summary.json"
    generations_path="$out_dir/generations.jsonl"
    log_path="$LOG_ROOT/${model_stem}-r${ROLLOUT}-seed${SEED}.log"

    if is_complete_run "$summary_path" "$generations_path" "$filename"; then
        printf '[%d/%d] Already complete; skipping %s\n' "$index" "$model_count" "$filename"
        continue
    fi

    if [[ ! -s "$model_path" ]]; then
        available_bytes="$(df -B1 --output=avail "$MODEL_DIR" | tail -n 1 | tr -d '[:space:]')"
        [[ "$available_bytes" =~ ^[0-9]+$ ]] || die "Could not read free space for $MODEL_DIR"
        required_bytes=$((reserve_gib * 1024 * 1024 * 1024))
        if (( available_bytes < required_bytes )); then
            die "Not enough space in $MODEL_DIR for $filename: need about ${reserve_gib} GiB free, have $((available_bytes / 1024 / 1024 / 1024)) GiB"
        fi
    fi

    printf '\n[%d/%d] Downloading %s (bsz=%d)\n' "$index" "$model_count" "$filename" "$bsz"
    if ! hf download "$uri" --local-dir "$MODEL_DIR"; then
        die "Download failed; any partial checkpoint is left in $MODEL_DIR"
    fi
    [[ -s "$model_path" ]] || die "Download completed without creating $model_path"

    printf '[%d/%d] Evaluating %s\n' "$index" "$model_count" "$filename"
    if "$PYTHON_BIN" "$SCRIPT_DIR/eval_aime24.py" \
        --model "$model_path" \
        --rollout "$ROLLOUT" \
        --bsz "$bsz" \
        --ks 1,2,4,8,16,32 \
        --seed "$SEED" \
        --gpus "$AIME_GPUS" \
        --out-dir "$out_dir" 2>&1 | tee "$log_path"; then
        :
    else
        eval_status=$?
        printf 'Evaluation failed with status %d; checkpoint retained at %s\n' \
            "$eval_status" "$model_path" >&2
        exit "$eval_status"
    fi

    if ! is_complete_run "$summary_path" "$generations_path" "$filename"; then
        die "Result verification failed; checkpoint retained at $model_path"
    fi

    rm -- "$model_path"
    printf '[%d/%d] Verified and removed checkpoint; results: %s\n' \
        "$index" "$model_count" "$summary_path"
done

printf '\nAll listed AIME24 rollout-%d evaluations are complete.\n' "$ROLLOUT"
