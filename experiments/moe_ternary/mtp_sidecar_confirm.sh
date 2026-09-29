#!/usr/bin/env bash
# End-to-end confirmation of the Scion k=1 MTP sidecar drafter (release drafter
# lane).  Runs against the actual released model + sidecar and gates on the
# measured numbers, so a regression fails loudly instead of silently.
#
# Stage 1  teacher-forced acceptance on the probe windows (fork test)
# Stage 2  real greedy generation with the runtime inputs (fork test, gen-api)
# Stage 3  llama-server end-to-end: baseline vs sidecar (acceptance, tok/s)
#
# Artifacts: $MOE/qwen35/mtp-sidecar-confirm/report.json + stage logs.
# Exit: 0 = all gates pass, 1 = a gate failed, 2 = prerequisite/run failure.
#
# Prereqs (all local): the fork build's test-mtp-sidecar + llama-server, the
# release GGUF at its verified sha256, and mtp-sidecar-v2.gguf.

set -uo pipefail

ROOT=/home/penis/Desktop/work/hivebench
MOE=$ROOT/artifacts/ternary/moe
Q=$MOE/qwen35
FORK=/home/penis/llama.cpp
BIN=$FORK/build/bin
MODEL=/home/penis/Desktop/qwen35-release.gguf
SIDECAR=$Q/mtp-sidecar-v2.gguf
PROBE=$Q/mtp-release
OUT=$Q/mtp-sidecar-confirm
RELEASE_SHA=545d83a93c6ad1474a97201a687f9a344df3e2d5f89ae7cc324aa3680be58d9c
PY=$ROOT/.venv/bin/python
GPU=${HIP_VISIBLE_DEVICES:-1}
PORT=${PORT:-8137}
PROMPT=${PROMPT:-"The history of science is the study of the development of"}
N_PREDICT=${N_PREDICT:-160}
TF_FW_WINDOWS=${TF_FW_WINDOWS:-16}   # 3056..3071 (fineweb held-out)
TF_WT_WINDOWS=${TF_WT_WINDOWS:-16}   # 3072..3087 (wikitext eval)
GEN_WINDOWS=${GEN_WINDOWS:-8}

mkdir -p "$OUT"
LOG="$OUT/confirm-$(date +%Y%m%d-%H%M%S).log"
REPORT="$OUT/report.json"
exec > >(tee -a "$LOG") 2>&1

say()  { printf '\n== %s\n' "$*"; }
fail_prereq() { printf 'PREREQ FAIL: %s\n' "$*"; exit 2; }

# gates (conservative vs the measured 2026-09-29 numbers, see §3c)
GATE_TF_FW=0.45; GATE_TF_WT=0.33
GATE_GEN_FW_ACC=0.55; GATE_GEN_WT_ACC=0.42
GATE_GEN_FW_TPF=1.50; GATE_GEN_WT_TPF=1.40
GATE_SRV_ACC=0.40; GATE_SRV_SPEEDUP=1.05

# ---------------------------------------------------------------- prereqs ---
say "prereqs"
[ -x "$BIN/test-mtp-sidecar" ] || fail_prereq "missing $BIN/test-mtp-sidecar (build the fork)"
[ -x "$BIN/llama-server" ]     || fail_prereq "missing $BIN/llama-server"
[ -f "$MODEL" ]                || fail_prereq "missing release GGUF $MODEL"
[ -f "$SIDECAR" ]              || fail_prereq "missing sidecar $SIDECAR"
[ -f "$PROBE/tokens.bin" ]     || fail_prereq "missing probe dir $PROBE"
if [ -d "$Q/.stage-lock" ]; then
    fail_prereq "a phase1 stage is running (.stage-lock exists)"
fi

say "release sha256"
GOT_SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
echo "expected $RELEASE_SHA"
echo "got      $GOT_SHA"
[ "$GOT_SHA" = "$RELEASE_SHA" ] || fail_prereq "release GGUF hash mismatch"

echo "sidecar: $(stat -c%s "$SIDECAR") bytes"
echo "started: $(date -Is)"

# ------------------------------------------------- stage 1: teacher-forced ---
say "stage 1 — teacher-forced acceptance (probe windows)"
TF_FW_LOG=$OUT/stage1-fineweb.log
TF_WT_LOG=$OUT/stage1-wikitext.log
HIP_VISIBLE_DEVICES=$GPU "$BIN/test-mtp-sidecar" "$SIDECAR" "$MODEL" "$PROBE" \
    3056 $((3056 + TF_FW_WINDOWS)) 512 gpu >"$TF_FW_LOG" 2>&1
TF_FW_RC=$?
HIP_VISIBLE_DEVICES=$GPU "$BIN/test-mtp-sidecar" "$SIDECAR" "$MODEL" "$PROBE" \
    3072 $((3072 + TF_WT_WINDOWS)) 512 gpu >"$TF_WT_LOG" 2>&1
TF_WT_RC=$?
TF_FW=$(grep -oE 'acceptance [0-9.]+' "$TF_FW_LOG" | awk '{print $2}' | tail -1)
TF_WT=$(grep -oE 'acceptance [0-9.]+' "$TF_WT_LOG" | awk '{print $2}' | tail -1)
echo "fineweb   rc=$TF_FW_RC acceptance=$TF_FW"
echo "wikitext  rc=$TF_WT_RC acceptance=$TF_WT"

# --------------------------------------------------- stage 2: generation ----
say "stage 2 — real generation (gen-api, exact runtime inputs)"
GEN_FW_LOG=$OUT/stage2-fineweb.log
GEN_WT_LOG=$OUT/stage2-wikitext.log
HIP_VISIBLE_DEVICES=$GPU "$BIN/test-mtp-sidecar" "$SIDECAR" "$MODEL" "$PROBE" \
    3056 $((3056 + GEN_WINDOWS)) 512 gpu gen-api >"$GEN_FW_LOG" 2>&1
GEN_FW_RC=$?
HIP_VISIBLE_DEVICES=$GPU "$BIN/test-mtp-sidecar" "$SIDECAR" "$MODEL" "$PROBE" \
    3072 $((3072 + GEN_WINDOWS)) 512 gpu gen-api >"$GEN_WT_LOG" 2>&1
GEN_WT_RC=$?

num() { grep -oE "$2" "$1" | grep -oE '[0-9.]+' | tail -1; }
GEN_FW_ACC=$(num "$GEN_FW_LOG" 'accepted \([0-9.]+\)')
GEN_FW_TPF=$(num "$GEN_FW_LOG" '= [0-9.]+ tokens/forward')
GEN_FW_DMS=$(num "$GEN_FW_LOG" 'draft [0-9.]+ ms/eval')
GEN_FW_FMS=$(num "$GEN_FW_LOG" 'forward [0-9.]+ ms/forward')
GEN_WT_ACC=$(num "$GEN_WT_LOG" 'accepted \([0-9.]+\)')
GEN_WT_TPF=$(num "$GEN_WT_LOG" '= [0-9.]+ tokens/forward')
GEN_WT_DMS=$(num "$GEN_WT_LOG" 'draft [0-9.]+ ms/eval')
GEN_WT_FMS=$(num "$GEN_WT_LOG" 'forward [0-9.]+ ms/forward')
echo "fineweb  rc=$GEN_FW_RC accept=$GEN_FW_ACC tokens/fwd=$GEN_FW_TPF draft=$GEN_FW_DMS ms fwd=$GEN_FW_FMS ms"
echo "wikitext rc=$GEN_WT_RC accept=$GEN_WT_ACC tokens/fwd=$GEN_WT_TPF draft=$GEN_WT_DMS ms fwd=$GEN_WT_FMS ms"

# ------------------------------------------------- stage 3: llama-server ----
say "stage 3 — llama-server baseline vs sidecar"

server_pid() { pgrep -f "llama-server.*--port $PORT" | head -1; }
server_stop() {
    local p
    p=$(server_pid)
    if [ -n "$p" ]; then
        kill "$p" 2>/dev/null
        for _ in $(seq 1 20); do
            [ -z "$(server_pid)" ] && break
            sleep 1
        done
        kill -9 "$(server_pid)" 2>/dev/null
    fi
}
trap server_stop EXIT INT TERM

server_start() { # $1 = extra args, $2 = log file; prints "ok" or the failure
    local extra="$1" log="$2"
    server_stop
    HIP_VISIBLE_DEVICES=$GPU setsid nohup "$BIN/llama-server" \
        -m "$MODEL" -c 2048 -ngl 999 -np 1 --port "$PORT" --host 127.0.0.1 \
        $extra >"$log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 120); do
        if grep -q 'listening on' "$log" 2>/dev/null; then
            echo "ok"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "died"
            return 1
        fi
        sleep 1
    done
    echo "timeout"
    return 1
}

srv_request() { # $1 = n_predict, $2 = cache_prompt, $3 = optional response file
    if [ -n "${3:-}" ]; then
        curl -s -m 300 "http://127.0.0.1:$PORT/completion" \
            -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$1,\"temperature\":0.0,\"cache_prompt\":$2,\"return_tokens\":true}" \
            >"$3"
    else
        curl -s -m 300 "http://127.0.0.1:$PORT/completion" \
            -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$PROMPT\",\"n_predict\":$1,\"temperature\":0.0,\"cache_prompt\":$2}" \
            >/dev/null
    fi
}

# first eval-time line, excluding the prompt-eval line; ms per token
ms_per_token() {
    grep 'eval time =' "$1" | grep -v 'prompt eval time' | head -1 \
        | sed -n 's/.*( *\([0-9.]*\) ms per token.*/\1/p'
}
srv_accept()  { grep -oE 'draft acceptance = [0-9.]+' "$1" | grep -oE '[0-9.]+$' | head -1; }
srv_meanlen() { grep -oE 'mean len = +[0-9.]+' "$1" | grep -oE '[0-9.]+$' | head -1; }

BASE_LOG=$OUT/stage3-baseline.log
R=$(server_start "" "$BASE_LOG"); [ "$R" = "ok" ] || fail_prereq "baseline server failed ($R)"
srv_request "$N_PREDICT" false "$OUT/stage3-baseline.json"
srv_request "$N_PREDICT" true    # cached prompt (baseline reference point)
grep -q 'eval time' "$BASE_LOG" || fail_prereq "baseline produced no eval timing"
BASE_MS=$(ms_per_token "$BASE_LOG")
echo "baseline: $BASE_MS ms/token"
server_stop

SPEC_LOG=$OUT/stage3-sidecar.log
R=$(server_start "-md $SIDECAR" "$SPEC_LOG")
[ "$R" = "ok" ] || fail_prereq "sidecar server failed ($R)"
srv_request "$N_PREDICT" false "$OUT/stage3-sidecar.json"
srv_request "$N_PREDICT" true    # cached prompt: exercises the no-ctx_dft save/load path
grep -q 'draft acceptance' "$SPEC_LOG" || fail_prereq "sidecar server produced no acceptance"
SRV_ACC=$(srv_accept "$SPEC_LOG")
SRV_LEN=$(srv_meanlen "$SPEC_LOG")
SPEC_MS=$(ms_per_token "$SPEC_LOG")
SPEEDUP=$(awk -v b="$BASE_MS" -v s="$SPEC_MS" 'BEGIN { printf "%.3f", (s > 0 ? b/s : 0) }')
echo "sidecar: $SPEC_MS ms/token, acceptance $SRV_ACC, mean len $SRV_LEN, speedup ${SPEEDUP}x"
server_stop

# diagnostic only: the runtime's multi-row verify batch is not bit-identical to
# a regular 1-row decode for this hybrid model, so greedy sequences can flip on
# near-ties (the dspark harness reports the same as ORACLE mismatches)
GREEDY_DIV=$("$PY" - "$OUT/stage3-baseline.json" "$OUT/stage3-sidecar.json" <<'EOF'
import json, sys
try:
    b = json.load(open(sys.argv[1])).get("tokens", [])
    s = json.load(open(sys.argv[2])).get("tokens", [])
except Exception:
    print(-1); raise SystemExit
for i, (x, y) in enumerate(zip(b, s)):
    if x != y:
        print(i); break
else:
    print(len(b) if b == s else -1)
EOF
)
echo "greedy token prefix: first divergence at index $GREEDY_DIV (diagnostic, not gated)"

for v in TF_FW TF_WT GEN_FW_ACC GEN_FW_TPF GEN_FW_DMS GEN_FW_FMS \
         GEN_WT_ACC GEN_WT_TPF GEN_WT_DMS GEN_WT_FMS BASE_MS SRV_ACC SRV_LEN SPEC_MS; do
    [ -n "${!v}" ] || fail_prereq "missing value for $v (a stage failed; logs in $OUT)"
done

# ----------------------------------------------------------------- gates ----
say "gates"
GATES_OK=1
gate() { # name value threshold
    local ok
    ok=$(awk -v v="$2" -v t="$3" 'BEGIN { print (v+0 >= t+0) ? 1 : 0 }')
    if [ "$ok" = "1" ]; then
        printf 'PASS  %-22s %s (gate >= %s)\n' "$1" "$2" "$3"
    else
        printf 'FAIL  %-22s %s (gate >= %s)\n' "$1" "$2" "$3"
        GATES_OK=0
    fi
}
gate tf_fineweb       "$TF_FW"      "$GATE_TF_FW"
gate tf_wikitext      "$TF_WT"      "$GATE_TF_WT"
gate gen_fineweb_acc  "$GEN_FW_ACC" "$GATE_GEN_FW_ACC"
gate gen_wikitext_acc "$GEN_WT_ACC" "$GATE_GEN_WT_ACC"
gate gen_fineweb_tpf  "$GEN_FW_TPF" "$GATE_GEN_FW_TPF"
gate gen_wikitext_tpf "$GEN_WT_TPF" "$GATE_GEN_WT_TPF"
gate server_accept    "$SRV_ACC"    "$GATE_SRV_ACC"
gate server_speedup   "$SPEEDUP"    "$GATE_SRV_SPEEDUP"

STATUS=OK; [ "$GATES_OK" = "1" ] || STATUS=FAIL

"$PY" - "$REPORT" <<EOF
import json, sys, datetime
out = sys.argv[1]
report = {
    "run": "mtp-sidecar-confirmation",
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "release_gguf": "$MODEL",
    "release_sha256": "$GOT_SHA",
    "sidecar": "$SIDECAR",
    "teacher_forced": {"fineweb": $TF_FW, "wikitext": $TF_WT},
    "generation": {
        "fineweb":  {"accept": $GEN_FW_ACC, "tokens_per_forward": $GEN_FW_TPF,
                     "draft_ms": $GEN_FW_DMS, "forward_ms": $GEN_FW_FMS},
        "wikitext": {"accept": $GEN_WT_ACC, "tokens_per_forward": $GEN_WT_TPF,
                     "draft_ms": $GEN_WT_DMS, "forward_ms": $GEN_WT_FMS}
    },
    "server": {"baseline_ms_per_token": $BASE_MS, "sidecar_ms_per_token": $SPEC_MS,
               "speedup": $SPEEDUP, "acceptance": $SRV_ACC, "mean_len": $SRV_LEN,
               "greedy_first_divergence": $GREEDY_DIV},
    "gates": {"tf_fineweb": $GATE_TF_FW, "tf_wikitext": $GATE_TF_WT,
              "gen_fineweb_acc": $GATE_GEN_FW_ACC, "gen_wikitext_acc": $GATE_GEN_WT_ACC,
              "gen_fineweb_tpf": $GATE_GEN_FW_TPF, "gen_wikitext_tpf": $GATE_GEN_WT_TPF,
              "server_accept": $GATE_SRV_ACC, "server_speedup": $GATE_SRV_SPEEDUP},
    "status": "$STATUS"
}
with open(out, "w") as f:
    json.dump(report, f, indent=2)
    f.write("\n")
print("wrote", out)
EOF

say "summary"
echo "teacher-forced: fineweb $TF_FW, wikitext $TF_WT"
echo "generation    : fineweb $GEN_FW_ACC acc / $GEN_FW_TPF tok/fwd; wikitext $GEN_WT_ACC / $GEN_WT_TPF"
echo "server        : $SPEC_MS ms/token vs baseline $BASE_MS (${SPEEDUP}x), accept $SRV_ACC, mean len $SRV_LEN, greedy prefix diverge @ $GREEDY_DIV"
echo "status        : $STATUS  (log: $LOG)"
[ "$STATUS" = "OK" ] || exit 1
exit 0
