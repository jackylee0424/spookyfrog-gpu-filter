#!/usr/bin/env bash
# Build the per-shot evaluator that tools/audit.py compares the prefilter against.
#
#   bash tools/make_evaluator.sh CHECKOUT            in place: adds CHECKOUT/src/bin/eval_paired.rs and builds there
#   bash tools/make_evaluator.sh CHECKOUT OUT_DIR    in a copy of CHECKOUT (without target/ and .git); CHECKOUT untouched
#
# eval_paired.rs = the stock src/bin/eval_circuit.rs + tools/eval_paired.patch. The patch only prints one FAILSHOT
# line per classically wrong shot (plus PHASEBATCH / ANCBATCH lines) and reads the stream from $EVAL_OPS; the checks,
# the shot draws and the counts are the stock ones. The new file sits outside the benchmark's editable path
# (src/point_add), so it never ships with a submission. Prints the evaluator path on its last line.
# CARGO_FLAGS is passed to cargo (for example CARGO_FLAGS="--offline --locked").
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
[ $# -eq 1 ] || [ $# -eq 2 ] || { echo "usage: bash tools/make_evaluator.sh CHECKOUT [OUT_DIR]"; exit 2; }
co="$(cd "$1" && pwd)"
[ -f "$co/src/bin/eval_circuit.rs" ] || { echo "$co is not a benchmark checkout (no src/bin/eval_circuit.rs)"; exit 2; }
if [ $# -eq 2 ]; then
  out="$2"
  [ -e "$out" ] && { echo "OUT_DIR already exists: $out"; exit 2; }
  mkdir -p "$out"
  tar -C "$co" --exclude=./target --exclude=./.git -cf - . | tar -C "$out" -xf -
  out="$(cd "$out" && pwd)"
else
  out="$co"
fi
cp "$out/src/bin/eval_circuit.rs" "$out/src/bin/eval_paired.rs.new"
patch --quiet "$out/src/bin/eval_paired.rs.new" < "$here/eval_paired.patch"
mv "$out/src/bin/eval_paired.rs.new" "$out/src/bin/eval_paired.rs"
(cd "$out" && cargo build --release ${CARGO_FLAGS:-} --bin eval_paired --bin eval_circuit)
echo "evaluator: $out/target/release/eval_paired"
