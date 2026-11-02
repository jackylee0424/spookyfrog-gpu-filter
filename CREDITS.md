# Credits

## Authors

Built by the spookyfrog team for the ecdsa.fail benchmark, with coding agents:

- Claude Opus 5.5 (xhigh), in Claude Code
- GPT-6.1 Sol (xhigh), in Codex CLI

## Code in this repository

- **Scanner core: John Boyer (johnbxx on the board).** The SHAKE256 prefix checkpoint and nonce-tail patch, the secp256k1 field arithmetic,
  the Jacobian point code, the fixed-base comb, the warp-wide batch inversion and the host driver of
  `src/leapfrog_filter.cu` derive from John Boyer's classical "ping-pong" tail-nonce prefilter, a Rust CPU reference
  (`pingpong_filter.rs`) and a CUDA port, published at https://github.com/somejohnbforya/pingpong-prefilter (commit
  `da0e572`, 2026-08-22). The `pingpong_filter.rs:N` comments in `src/leapfrog_filter.cu` cite the lines of that
  reference. That repository carries no licence file, so the parts of `src/leapfrog_filter.cu` derived from it are not
  offered under this repository's MIT licence unless its author grants terms.
- **Walk model: the spookyfrog team.** The classical models of the divide and multiply walks (`src/lf_walk.cuh`,
  `src/sf43h_hint_flat.cuh`, `tools/lfmodel.py`), the run-time binding and rebind tools, the fidelity audit and the
  choice policy (`src/fa50_policy.cuh`), built from October 6, 2026 onward (`docs/HISTORY.md`).
- **Benchmark: Layr-Labs.** The harness, evaluator and stream format are those of the ecdsa.fail challenge repository
  (https://github.com/Layr-Labs/ecdsafail-challenge). `tools/eval_paired.patch` adds per-shot output lines to a local
  copy of its evaluator; the patch is applied to your own checkout and changes no check.

## Prior work on nonce prefilters

This filter stands on earlier public work on screening tail nonces before the full evaluation. The earliest instances we
found in the public submission notes of the board:

- **Jieyi Long (jieyilong)**: the first bit-exact CUDA port of a circuit's classical prefilter (submission 98495351,
  2026-06-06; a block-per-nonce, shot-parallel kernel on two A100s), then the public, MIT-licensed
  https://github.com/jieyilong/ecdsafail_gpu_toolkit (submission ff46db24, 2026-06-07). It exports the fixed SHAKE
  prefix, the circuit's schedules and the secp256k1 comb table once, then scans nonce tails on the GPU. This filter
  follows the same design; no code from the toolkit is in this repository.
- **John Boyer (johnbxx)**: a classical convergence prefilter feeding a GPU compare and width screen (64f75ba9,
  2026-06-05), and later the ping-pong prefilter that this repository's scanner core comes from (54ff37e6 and
  ed30a235, 2026-08-22).
- **newjordan**: classical prefilter lanes on rented CPU machines that found clean nonces for new op streams (012fba1f
  and following entries, 2026-06-06).
- **alexander-sei**: a classical convergence filter (488afae5, 2026-06-05), which later notes credit; and
  **PhantasticUniverse**: a width and convergence prefilter in front of the full simulator (b29ff7ff, 2026-06-04).

## What the walk model follows

The filter reproduces, on classical values, the walks that the baseline's own source builds. These ideas are theirs:

- **Leapfrog**, the divide and multiply walk family: **BitWonka** (submission 86221ad).
- **The four-bit tick-0 equal-sign veto window (KH4)**: **haotang1995** (01a5cc24).
- **The choice-compare narrowing and the last-tick constant veto**: **haotang1995** (dca38d60); the deepened
  choice-narrowing rows at the most deeply narrowed ticks: **odinfree** (98460d66). `src/fa50_policy.cuh` encodes the
  rows that this lineage emits.
- **The deeper walk and the tick-0 carry identity**: **zuiris** (4cb9cf01); its composition with the spookyfrog square
  and the record-slot, k3, reverse-tick and fold-ladder cuts: **gnuchev** (88911269, 427ac4c9).

## The baseline circuit and its lineage

The baseline, spookyfrog 24 (submission 859d928d), adds one exact table to **odinfree**'s 98460d66. Its tree carries
the work of, by board entry:

- **odinfree**: 98460d66, f20f4d87 (a sibling branch, not carried), 57f98ac5 (`cmp_narrow`), 7d81445b and db0c5ec7
  (lane R);
- **haotang1995**: dca38d60 (tapers, cuts, caps, per-block widths, choice narrowing, vetoes, `cmp_extra`, top-AND cuts,
  `hold_pass`) and 01a5cc24 (KH4);
- **osipovartem**: the row certificates of research submission 2d8e5f14;
- **Th0rgal (Maxwell on Orb)**: 9ef3139d, the swap-hold table with relays;
- **Champollion9012**: the census tables of 46ae05c5 and 076f1100;
- **mochimodev (Matt Zweil and MAI-A.A.E., Mochimo Foundation)**: a362dcaa, c1b95540, ea64e5cb, 8a1da769 and
  fa27526a, the tree under all the tables; and Skywalk and U4;
- **gnuchev**: 88911269 and 427ac4c9;
- **zuiris**: 4cb9cf01;
- **Meganpark980320**: fe0ced81, the swap-park hold form;
- **MonitSharma**: d7f69382 and d9ae2845, the AND cache and the general classical passes;
- **kaankolcu**: 767d8d5a, 7927f4b0 and 4daa9b1e, the narrowed compares and census cuts;
- **jackylee0424**: 9da9a45d, a9b17310, e4075d6c and earlier entries, the rail-top tables, the zero-fill rail frame and
  the cache generator;
- **krishparadox**: 8f980ccb, d4481580 and earlier entries;
- **BitWonka**: Leapfrog;
- **Craig Gidney**: measurement-based uncomputation and deferred phase fixes (arXiv 2505.15917, section 2.4);
- **the spookyfrog team**: spookyfrog 19 to 24 (be1f667d, da097c38, ee656d09, 45adc970, 859d928d).

The baseline's public submission note (`ecdsafail submission-note 859d928d`) gives each contribution in full. It also
carries, verbatim, the limiting contracts of mochimodev's note for 8ee337c and of PR 419, and states that the
baseline's inherited failure rate (about 21.2 failing shots per 9,024) is above both limits.

## Measurements

Every number in this repository was measured on one RTX 5090 (`sm_120`) under WSL2 unless stated otherwise. See
`VERIFY.json` and `docs/HISTORY.md`.

## Corrections

If your work is used here and is missing or described wrongly, please open an issue.
