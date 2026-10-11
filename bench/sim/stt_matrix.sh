#!/usr/bin/env bash
# STT on grace-sim output: each scene set × signal × decoder, oracle cuts. Results append to
# bench-results/stt-sim/summary.jsonl (label = <prefix>-<set>-<signal>-<decoder>).
#   bash bench/sim/stt_matrix.sh full "near far doubletalk"
set -u
prefix=$1; sets=$2
py=bench/.venv/Scripts/python.exe
declare -A dec=(
  [prod]="$py bench/runners/stt_faster_whisper.py --model danish-foundation-models/edda-v0.2 --compute int8_float16"
  [fp16b5]="$py bench/runners/stt_faster_whisper.py --model danish-foundation-models/edda-v0.2 --compute float16 --beams 5 --no-fallback"
  [ekko]="$py bench/runners/stt_sherpa_transducer.py --repo RyeAI/ekko-v1-tiny --subdir onnx-sherpa --int8"
)
for set in $sets; do
  for sig in raw lin aec; do
    ./target/release/grace-bench.exe stt --manifest bench-results/sim/$prefix-$set/oracle-$sig.jsonl \
      --label $prefix-$set-$sig-prod --out bench-results/stt-sim -- ${dec[prod]} >/dev/null 2>&1
  done
  for d in fp16b5 ekko; do
    ./target/release/grace-bench.exe stt --manifest bench-results/sim/$prefix-$set/oracle-lin.jsonl \
      --label $prefix-$set-lin-$d --out bench-results/stt-sim -- ${dec[$d]} >/dev/null 2>&1
  done
done
