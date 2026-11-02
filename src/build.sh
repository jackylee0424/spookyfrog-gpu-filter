#!/usr/bin/env bash
# Build the Leapfrog GPU prefilter in this directory (writes ./leapfrog_filter). Starts no scan.
#
#   ARCH        GPU architecture, default sm_120 (RTX 50-series). Others: sm_89 (RTX 40-series), sm_86 (RTX 30-series,
#               A10/A40), sm_80 (A100), sm_90 (H100). Use the value `nvidia-smi --query-gpu=compute_cap` reports
#               (8.9 -> sm_89).
#   MINBLOCKS   occupancy knob of the point kernel (minBlocksPerMultiprocessor), default 3, tuned on an RTX 5090.
#               Try 2 on cards with fewer registers per SM if ptxas reports spills.
#   COMB_BITS   fixed-base comb window, 8..16, default 16 (the 16-bit comb table lives in GPU memory; 12 or 14 use less).
#   NVCC        compiler, default the nvcc on PATH, else /usr/local/cuda/bin/nvcc.
#
# Any rebind (see docs/REBIND.md) changes the compiled-in checkpoint and op count: rebuild afterwards, then rerun the
# fidelity audit before trusting the binary.
set -euo pipefail
cd "$(dirname "$0")"
NVCC="${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
ARCH="${ARCH:-sm_120}"
MINBLOCKS="${MINBLOCKS:-3}"
COMB_BITS="${COMB_BITS:-16}"
"$NVCC" -O3 -std=c++17 -DLF_YP8_VETO=1 -DFE_FAST_MUL=1 -DCOMB_BITS="$COMB_BITS" -DPOINT_MINBLOCKS="$MINBLOCKS" \
  -arch="$ARCH" -Xptxas=-v leapfrog_filter.cu -o leapfrog_filter
echo "built $(pwd)/leapfrog_filter for $ARCH"
