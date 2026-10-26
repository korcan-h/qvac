#!/usr/bin/env bash
# TEMP (s26 Vulkan diagnosis): runs on the Device Farm host in pre_test, before Appium starts.
# Pushes the diagnostic ggml/llama tools to the phone, runs them over adb against the prestaged
# Qwen3.5 model, and leaves every output in $DEVICEFARM_LOG_DIR/qvac-diag. It never fails the
# test spec: every step has a device-side timeout, and the whole script keeps to a time budget.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${DEVICEFARM_LOG_DIR:-/tmp}/qvac-diag"
D=/data/local/tmp/qvac-diag
M=/data/local/tmp/prestaged-models/Qwen3.5-0.8B-Q4_0.gguf
BUDGET="${QVAC_DIAG_BUDGET:-2700}"
T0=$SECONDS
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"

log() { echo "[qvac-diag] $*" | tee -a "$SUMMARY"; }

# run <name> <timeout s> <device shell command...>
run() {
    local name=$1 t=$2; shift 2
    local left=$(( BUDGET - (SECONDS - T0) ))
    if [ "$left" -lt 60 ]; then
        log "$name: skipped (budget used up)"
        return 0
    fi
    [ "$t" -gt "$left" ] && t=$left
    local start=$SECONDS
    echo "=== $name (timeout ${t}s): $*" > "$OUT/$name.txt"
    timeout -k 15 $(( t + 30 )) adb shell "cd $D && export LD_LIBRARY_PATH=$D/lib && timeout $t sh -c '$*'" >> "$OUT/$name.txt" 2>&1
    local rc=$?
    log "$name: rc=$rc in $(( SECONDS - start ))s"
}

log "start $(date -u +%FT%TZ) device=$(adb shell getprop ro.product.model | tr -d '\r') budget=${BUDGET}s"
adb shell getprop ro.build.fingerprint > "$OUT/device.txt" 2>&1
adb shell getprop ro.hardware.vulkan >> "$OUT/device.txt" 2>&1

adb shell rm -rf "$D"
adb shell mkdir -p "$D/lib"
adb push "$HERE/lib/." "$D/lib/" > "$OUT/push.txt" 2>&1
adb push "$HERE/bin/." "$D/" >> "$OUT/push.txt" 2>&1
adb push "$HERE/prompt.txt" "$HERE/train.jsonl" "$D/" >> "$OUT/push.txt" 2>&1
adb shell "chmod 755 $D/test-* $D/llama-*"
if ! adb shell test -s "$M"; then
    log "model $M missing on device; model steps will fail"
fi

# 1. Device summary and the diag line (arch, robustness, coopmat, int dot, fp16, subgroups)
run info 120 "./test-backend-ops -b Vulkan0 -o NO_SUCH_OP"
run info-keep-robustness 120 "GGML_VK_KEEP_ROBUSTNESS=1 ./test-backend-ops -b Vulkan0 -o NO_SUCH_OP"

# 2. Every unique op of the real forward graph (ctx = ubatch = 128, no flash attention, as in the
#    failing finetune), tested on Vulkan against the CPU
run export 300 "./test-export-graph-ops -m $M -c 128 -b 128 -ub 128 -fa off -o $D/qwen35-ops.txt"
adb pull "$D/qwen35-ops.txt" "$OUT/" >> "$OUT/push.txt" 2>&1
run tbo-graph 900 "./test-backend-ops -b Vulkan0 --test-file $D/qwen35-ops.txt"

# 3. Per-tensor dump of the real model forward on CPU and Vulkan: diff the sums offline to find
#    the first node that diverges
EVAL="./llama-eval-callback -m $M -f $D/prompt.txt -c 256 -b 128 -ub 128 -fa off --no-warmup -t 6"
run evalcb-cpu 600 "$EVAL -dev none -ngl 0"
run evalcb-vk 600 "$EVAL -dev Vulkan0 -ngl 99"

# 4. The failing finetune itself (same data, LoRA modules and lr as runFinetuningArchsTest):
#    step losses on CPU, on Vulkan, and on Vulkan with each candidate switch
FT="./llama-finetune-lora -m $M -f $D/train.jsonl --assistant-loss-only -c 128 -b 128 -ub 128 -fa off --lora-modules attn_q,attn_k,attn_v,attn_o --learning-rate 1e-5 --lr-min 1e-8 --num-epochs 1 --checkpoint-save-steps 100000 --checkpoint-save-dir $D/ckpt --output-adapter $D/adapter.gguf -t 6"
run ft-vk 600 "$FT -dev Vulkan0 -ngl 99"
run ft-cpu 900 "$FT -dev none -ngl 0"
run ft-vk-keep-robustness 600 "GGML_VK_KEEP_ROBUSTNESS=1 $FT -dev Vulkan0 -ngl 99"
for v in DISABLE_INTEGER_DOT_PRODUCT DISABLE_F16 DISABLE_FUSION DISABLE_COOPMAT DISABLE_MMVQ DISABLE_GRAPH_OPTIMIZE PREFER_HOST_MEMORY; do
    run "ft-vk-$v" 600 "GGML_VK_$v=1 $FT -dev Vulkan0 -ngl 99"
done
run ft-vk-serialized 900 "GGML_VK_SERIALIZE_SUBMISSIONS=1 GGML_VK_MAX_NODES_PER_SUBMIT=1 $FT -dev Vulkan0 -ngl 99"

# 5. Robustness variants of the op-level checks
run tbo-graph-keep-robustness 900 "GGML_VK_KEEP_ROBUSTNESS=1 ./test-backend-ops -b Vulkan0 --test-file $D/qwen35-ops.txt"
run evalcb-vk-keep-robustness 600 "GGML_VK_KEEP_ROBUSTNESS=1 $EVAL -dev Vulkan0 -ngl 99"

# 6. The standard test suites of the ops this model uses, as far as the budget allows
run tbo-mul-mat 1200 "./test-backend-ops -b Vulkan0 -o MUL_MAT -p \"type_a=(f32|f16|q4_0|q4_1|q8_0|q5_K|q6_K),\""
run tbo-other-ops 1200 "./test-backend-ops -b Vulkan0 -o GATED_DELTA_NET,SSM_CONV,L2_NORM,RMS_NORM,NORM,ROPE,SOFT_MAX,SWIGLU,SILU,SIGMOID,SOFTPLUS,EXP,NEG,GET_ROWS,SET_ROWS,CPY,CONT,CONCAT,ADD,MUL,SCALE,SUM_ROWS,ACC"

# Short results in the host log too (Test_spec_output), in case the artifact folder is dropped
{
    echo "--- device summary"
    grep -ah "ggml_vulkan: [0-9] = \|ggml_vulkan: diag:" "$OUT"/info*.txt
    echo "--- test-backend-ops failures (first 40 per run)"
    for f in "$OUT"/tbo-*.txt; do
        echo "## $(basename "$f"): $(grep -ac 'FAIL' "$f") FAIL lines, $(grep -a 'tests passed' "$f" | tail -1)"
        grep -a 'FAIL' "$f" | head -40
    done
    echo "--- finetune losses"
    for f in "$OUT"/ft-*.txt; do
        echo "## $(basename "$f" .txt): $(grep -aoE 'loss=[0-9.na-]+' "$f" | head -6 | tr '\n' ' ')"
    done
} >> "$SUMMARY" 2>&1
log "done in $(( SECONDS - T0 ))s"
cat "$SUMMARY"

adb shell rm -rf "$D"
exit 0
