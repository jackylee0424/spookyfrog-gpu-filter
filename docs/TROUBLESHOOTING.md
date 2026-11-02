# Troubleshooting

Run `python3 tools/sfgf.py doctor` first; most problems show up there with a hint. Logs of every step are in
`work/logs/`.

## Building the filter

| symptom | cause | fix |
|---|---|---|
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | CUDA older than 12.8, or a card that is not RTX 50-series | `ARCH=sm_89 python3 tools/sfgf.py build` (RTX 40), `sm_86` (RTX 30, A10/A40), `sm_80` (A100), `sm_90` (H100); `doctor` prints `suggested_arch` from `nvidia-smi` |
| `ptxas` reports spills, or throughput far below the table in `docs/DESIGN.md` | occupancy tuned for an RTX 5090 | `MINBLOCKS=2 python3 tools/sfgf.py build`; on cards with little memory also `COMB_BITS=12` |
| `nvcc: command not found` | toolkit not on `PATH` | add `/usr/local/cuda/bin` to `PATH` or set `NVCC=/path/to/nvcc` |
| `cudaErrorMemoryAllocation` or out of memory at start | `--batch 65536` and the 16-bit comb table need a few GB | close other GPU jobs; rebuild with `COMB_BITS=12`; run `src/leapfrog_filter` by hand with a smaller `--batch` |

## Running the filter

| symptom | cause | fix |
|---|---|---|
| `REFUSING TO RUN: checkpoint bytes differ from the compiled frozen packet` | `src/checkpoint.bin` changed after the build (a rebind without a rebuild), or the wrong file was passed | `python3 tools/sfgf.py build` after every rebind |
| an op-count error, or `--ops <count> is REQUIRED` | `--ops` differs from `LF_EXPECTED_OPS` | use the count in `binding.json` `stream.ops`; the tools do this for you |
| `scan` refuses with "no passing fidelity audit" | this binary (by hash) has no audit of at least 48 packets | `python3 tools/sfgf.py audit`; every rebuild produces a new hash |
| `reference_windows_mismatched` not empty | the build or binding differs from the shipped one on the baseline stream | stop; rebuild from a clean clone; compare `binding.json` with `expected/spookyfrog24.json` |

## Harness and evaluator

| symptom | cause | fix |
|---|---|---|
| `pinned Rust toolchain '...' is not installed` | the benchmark pins a toolchain | run `./setup.sh` inside `work/baseline`, then `build-stream` again |
| `no sandbox available (bubblewrap/sandbox-exec)` warning | `bwrap` missing | harmless for replication (the harness runs `build_circuit` unconfined); install `bubblewrap` to sandbox it |
| `build-stream`: `match.raw_sha256` false but `match.prefix_raw_sha256` true | the baked nonce differs (someone edited `mod.rs`) | re-fetch the baseline |
| `build-stream`: prefix differs | wrong commit, a changed harness, or a different toolchain changed the emitted stream | stop and report the JSON; do not rebind to "fix" the baseline |
| `libzstd not found` | missing shared library | `sudo apt install zstd libzstd1` |
| evaluator build fails in `audit` | `cargo` offline mode without fetched crates | run `build-stream` first (it runs `cargo fetch --locked`) |

## Signs of false rejections (a wrong binding)

A binding that rejects clean nonces looks healthy: it runs, it is fast, it even passes a known-answer test. Watch for:

- the audit reports `native_good_gpu_bad > 0` (the definitive sign; the per-row `native_good_gpu_bad` lists the shots);
- a nonce known to be clean on the stream is rejected by `python3 tools/audit.py positive --nonce N`;
- zero survivors over tens of millions of nonces where the lineage normally keeps about 0.2 per million;
- the stream's source changed `choice_narrow.txt`, `hold_pass.rs` or anything before the hold pass since the binding
  was made (`check` lists the files).

The fix is the next rebind level (`docs/REBIND.md`), never a looser audit.

## WSL2 notes

- Install the NVIDIA driver on Windows (not inside WSL) and the CUDA toolkit for WSL inside the distribution.
- Keep `work/` on the Linux file system (clone the repository under your Linux home) for speed; the tools read
  multi-hundred-megabyte streams.
