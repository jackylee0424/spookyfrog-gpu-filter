# AGENTS.md: contract for coding agents

This file is the single source of instructions for any coding agent (Claude Code, Codex, others) working in this
repository. Read it fully before running anything. `README.md` is the human overview; `docs/` holds the details.

## Goal

Replicate the spookyfrog GPU nonce prefilter end to end on the ecdsa.fail baseline, then (optionally) use it to screen
nonces on that stream or on a stream you rebind it to.

The replication target is mechanical: on the baseline stream the filter must pass its fidelity audit, and the known
clean nonce `1453896738487` must survive the filter and score exactly `912499810` (1,234 qubits x 739,465 Toffoli)
with the stock evaluator.

The baseline is the current best promoted ecdsa.fail submission: **spookyfrog 24**, submission
`859d928d-76ee-4340-a466-ce6bce98f1c7`, public commit `39b7cd573f5a7a6b01970a9cd4f11a462e398ea1` of
`https://github.com/Layr-Labs/ecdsafail-challenge`. All expected values live in `expected/spookyfrog24.json`.

## Prerequisites

| need | why | check |
|---|---|---|
| Linux or WSL2 | the harness and the tools assume Linux | `doctor` |
| NVIDIA GPU + driver, CUDA toolkit with `nvcc` (12.8+ for RTX 50-series `sm_120`) | the filter is CUDA | `doctor` reports the card and the `ARCH` to use |
| Rust toolchain (`cargo`, `rustc`) and `gcc` | the benchmark harness and `tools/lftools` are Rust | `doctor` |
| Python 3.9+ (standard library only), `git`, `patch`, `tar`, `libzstd` | the tools | `doctor` |
| about 6 GB free disk, network access to GitHub and crates.io | baseline checkout and its build | `doctor` |
| optional: `bwrap` (sandbox for the harness), the `ecdsafail` CLI | the CLI is only needed to submit | `doctor` |

No API key or account is needed for the replication path.

## Replication path

Run every command from the repository root. Each command ends with one line `SFGF <command> <json>` and exits
`0` = PASS, `1` = FAIL, `2` = setup or usage error. Check the exit code and the JSON fields named below. Work files go
to `./work/` (git-ignored).

| step | command | pass criteria | typical time |
|---|---|---|---|
| 0 | `python3 tools/sfgf.py doctor` | exit 0; `"missing": []`. Note `suggested_arch`. | seconds |
| 1 | `python3 tools/sfgf.py fetch-baseline` | exit 0; `head` == `expected` == `39b7cd57...` | under a minute |
| 2 | `python3 tools/sfgf.py build-stream` | exit 0; `match.ops`, `match.prefix_raw_sha256`, `match.raw_sha256`, `match.score` all `true`; `score` == `912499810` | 5-15 min (Rust build + one full evaluation) |
| 3 | `python3 tools/sfgf.py check` | exit 0; `level` == `0` on the baseline | seconds |
| 4 | `python3 tools/sfgf.py rebind --level 1` | exit 0; `matches_shipped_baseline_checkpoint` == `true` | under a minute |
| 5 | `python3 tools/sfgf.py build` | exit 0; prints `arch` and `binary_sha256` | 1-3 min |
| 6 | `python3 tools/sfgf.py audit` | exit 0; `native_good_gpu_bad` == `0`; `checker_control` == `true` | 5-15 min (first run also builds the per-shot evaluator) |
| 7 | `python3 tools/sfgf.py control` | exit 0; `replication` == `"PASS"`; `checks` all `true`; `score` == `912499810` | about 1 min |

Step 4 is optional on the baseline (step 3 says level 0, and the shipped binding already matches it), but running it
shows your rebind tooling reproduces the shipped checkpoint byte for byte. On any other stream, step 4 is where the
level from step 3 is applied (see "Rebinding" below).

After step 7 the replication is complete. Report the JSON lines of steps 2, 6 and 7.

## Using the filter (after a passing audit)

| step | command | pass criteria |
|---|---|---|
| 8 | `python3 tools/sfgf.py scan --from N --count M` | exit 0. Refuses to run unless the current binary has a passing audit of at least 48 packets. Survivors go to `work/scans/N/survivors.bin`. |
| 9 | `python3 tools/sfgf.py validate --survivors work/scans/N/survivors.bin` | each `VALIDATE` line has `clean`; the final JSON lists clean nonces with `score` |

A quick self-check of step 8: `scan --from 1453896000000 --count 1000000` must report
`reference_windows_compared: 1`, `reference_windows_mismatched: []`, and its survivors include `1453896738487`.

Expect about 0.23 survivors per million nonces on the baseline stream; in the original search 1 of 423 survivors was
fully clean (the filter does not see phase failures). On an RTX 5090 the scan runs at about 109,000 nonces/s.

## Rebinding to another stream

Build the other stream with that checkout's own `./benchmark.sh` first, then
`python3 tools/sfgf.py --dir PATH_TO_CHECKOUT check` reports the level (it compares the checkout's `src/point_add`
and its `ops.bin` against `binding.json`; every later command also takes `--dir`). Level 1 and level 3 are scripted (`rebind --level 1|3`); level 2 is a manual
procedure in `docs/REBIND.md`. After ANY rebind: `build`, then `audit`, then a positive control on a nonce you know is
clean on that stream if you have one. See `docs/REBIND.md`.

## Stop conditions

Stop and report (do not work around) when:

- `doctor` lists anything under `missing`;
- `build-stream` reports any `match` field `false` (wrong commit, changed harness, or a different toolchain changed the
  stream);
- `audit` reports `native_good_gpu_bad` > 0 or exits non-zero: the filter would reject clean nonces on this stream;
- `control` reports `replication: FAIL`;
- a scan's `reference_windows_mismatched` is not empty;
- any tool prints `REFUSING TO RUN` or an op-count mismatch (the binary is not bound to the stream you passed).

## Never

- Never skip the fidelity audit after a rebind or a rebuild for a new stream. A filter that passes a known-answer test
  can still reject most clean nonces; this happened once (`docs/HISTORY.md`).
- Never treat a survivor as a result. Only a stock-evaluator run that reports every shot OK (`validate`, `clean: true`)
  is a result, and only a rebuild of the source with that nonce (the replay in `docs/REBIND.md`) is a candidate.
- Never edit `src/fa50_policy.cuh`, `src/lf_*.cuh` or `src/checkpoint*` by hand; regenerate them with `tools/rebind.py`
  (or the level 2 procedure) and re-audit.
- Never submit a circuit that is someone else's work as your own. If you submit a nonce found with this tool on
  another team's circuit, say so in the submission note and credit the circuit's authors.
- Never commit `work/`, `ops.bin`, the compiled `src/leapfrog_filter` or `tools/lftools/target/` (`.gitignore` covers
  them).

## Repository map

| path | what |
|---|---|
| `tools/sfgf.py` | the driver above (one subcommand per step) |
| `tools/rebind.py` | rebind levels 1 and 3, the level check, the `binding.json` manifest |
| `tools/audit.py` | fidelity audit, positive screen, throughput |
| `tools/validate.py` | full stock-evaluator check of survivors; replay comparison |
| `tools/nonce.py` | rewrite / hash the 96-op nonce tail of an `ops.bin` |
| `tools/lftools/` | Rust helper: Fiat-Shamir prefix checkpoint (`ckpt`) and XOF words (`xof`) |
| `tools/lfmodel.py` | Python reference of the walk widths (used by level 3) |
| `tools/eval_paired.patch`, `tools/make_evaluator.sh` | per-shot listing copy of the stock evaluator for the audit |
| `src/` | the CUDA filter and its generated binding headers; `src/build.sh` |
| `binding.json` | what the shipped binding is bound to, and hashes of every file that affects behaviour |
| `expected/spookyfrog24.json` | expected values for the baseline (stream hashes, checkpoint, control nonce, reference windows) |
| `VERIFY.json` | the maintainers' own end-to-end run of this path from a fresh clone |
| `docs/` | `REBIND.md`, `DESIGN.md`, `TROUBLESHOOTING.md`, `HISTORY.md` |

## Conventions for changes to this repository

- Keep tools standard-library Python; keep every path relative to the repository root.
- After changing any file listed in `binding.json` `files`, run `python3 tools/rebind.py manifest`, then rerun steps
  5-7 and update `VERIFY.json` if behaviour could have changed.
