# Credits

## Authors

Built by the spookyfrog team for the ecdsa.fail benchmark, with coding agents:

- Claude Opus 5.5 (xhigh), in Claude Code
- GPT-6.1 Sol (xhigh), in Codex CLI

## Code lineage

- **Scanner core.** The SHAKE256 prefix checkpoint and nonce-tail patch, the secp256k1 field arithmetic, the Jacobian
  point code, the fixed-base comb, the warp-wide batch inversion and the host driver of `src/leapfrog_filter.cu`
  descend from John Boyer's classical "ping-pong" tail-nonce prefilter, a Rust CPU reference (`pingpong_filter.rs`)
  and a CUDA port, published at https://github.com/somejohnbforya/pingpong-prefilter (commit `da0e572`,
  2026-08-22). Comments in `src/leapfrog_filter.cu` still cite the lines of that reference. That repository carries
  no licence file, so the parts of `src/leapfrog_filter.cu` derived from it are not offered under this repository's
  MIT licence unless its author grants terms.
- **Walk model.** The classical model of the divide and multiply walks follows the Leapfrog circuit family introduced on
  the board by BitWonka (submission 86221ad, "Leapfrog") and the rails, choice and hold passes added by later public
  submissions on that lineage. The spookyfrog 24 circuit itself builds on public submissions by other participants; its
  submission note lists that lineage.
- **Benchmark.** The harness, evaluator and stream format are those of the ecdsa.fail challenge repository
  (`Layr-Labs/ecdsafail-challenge`). `tools/eval_paired.patch` adds per-shot output lines to a local copy of its
  evaluator; the patch is applied to your own checkout and changes no check.

## Measurements

Every number in this repository was measured on one RTX 5090 (`sm_120`) under WSL2 unless stated otherwise. See
`VERIFY.json` and `docs/HISTORY.md`.
