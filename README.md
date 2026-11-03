<p align="center">
  <img src="assets/logo.jpg" alt="spookyfrog logo: a frog wizard with a glowing staff among jack-o'-lanterns under an orange moon" width="320">
</p>

# spookyfrog-gpu-filter

A CUDA prefilter that screens Fiat-Shamir tail nonces for the ecdsa.fail secp256k1 point-addition benchmark, bound to
the Leapfrog-family circuit of the current best submission (**spookyfrog 24**: 1,234 qubits x 739,465 Toffoli =
912,499,810). It runs at about **109,000 nonces/s on one RTX 5090** and keeps about 0.23 nonces per million for the
full check.

> **It is a prefilter, not a grader.** It models the classical walk of the circuit's divide and multiply legs on every
> one of the 9,024 shots a nonce draws. It does not model phase garbage, approximate payload cells or ancilla cleanup.
> Every survivor still needs the stock evaluator (`tools/sfgf.py validate`). In the original search, 423 survivors
> gave 11 classically clean nonces and 1 fully clean nonce.

## Why it exists

The benchmark seeds its 9,024 test shots from a SHAKE256 hash of the whole op stream. The last 96 ops of the stream are
48 identity `X;X` pairs whose target qubit encodes a nonce, so every nonce gives an independent draw of shots and a
slightly different mean Toffoli count. Landing a clean, cheaper draw means trying many nonces. The stock path (build
plus evaluate) costs tens of seconds per nonce; this filter rejects almost all failing nonces on the GPU and leaves a
few hundred candidates per billion for the CPU.

## Quick start

Coding agents: follow [`AGENTS.md`](AGENTS.md); it is the contract for this repository.

People:

```bash
python3 tools/sfgf.py doctor           # what is installed, what is missing, which GPU arch to build for
python3 tools/sfgf.py fetch-baseline   # public benchmark repo at the spookyfrog 24 commit -> work/baseline
python3 tools/sfgf.py build-stream     # stock harness build + evaluation; must reproduce 912499810
python3 tools/sfgf.py check            # rebind level needed (0 on the baseline)
python3 tools/sfgf.py build            # compile the filter (ARCH auto-detected)
python3 tools/sfgf.py audit            # fidelity audit: 48 packets, must find 0 native-good/GPU-rejected shots
python3 tools/sfgf.py control          # the known clean nonce survives and scores 912499810
python3 tools/sfgf.py scan --from 1453896000000 --count 1000000   # reproduces a window of the original search
```

Every command ends with `SFGF <command> <json>` and exits 0 on PASS.

## Requirements

Linux or WSL2; an NVIDIA GPU with the CUDA toolkit (`nvcc`; CUDA 12.8 or newer for RTX 50-series); Rust and `gcc` (the
benchmark harness is Rust); Python 3.9+ standard library; `git`, `patch`, `tar`, `libzstd`. No account or API key is
needed to replicate.

## Using it on another circuit

The filter is bound to one op stream: a checkpoint of the hash prefix, the op count, the walk tables and a per-tick
choice policy extracted from the circuit's own operations. A new nonce on the same source needs nothing. A new
source needs a rebind at one of three levels, and **every rebind must be followed by the fidelity audit**. See
[`docs/REBIND.md`](docs/REBIND.md).

This filter models only the Leapfrog walks. For the dialog-GCD and TrailMix circuit families of June 2026, see Jieyi
Long's [ecdsafail_gpu_toolkit](https://github.com/jieyilong/ecdsafail_gpu_toolkit).

## Documentation

| file | contents |
|---|---|
| [`AGENTS.md`](AGENTS.md) | numbered replication path with pass criteria and stop conditions |
| [`docs/DESIGN.md`](docs/DESIGN.md) | how the filter works and what it does not model |
| [`docs/REBIND.md`](docs/REBIND.md) | rebind levels 1-3 and the mandatory audit |
| [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) | build flags, op-count and checkpoint errors, false-reject symptoms |
| [`docs/HISTORY.md`](docs/HISTORY.md) | lineage and the repair that took false rejections from 79 per 48 packets to 0 per 192 |
| [`CREDITS.md`](CREDITS.md) | authors, models used and code lineage |
| `VERIFY.json` | our own end-to-end run of the replication path from a fresh clone |

## Credits

- **Scanner core:** John Boyer's [pingpong-prefilter](https://github.com/somejohnbforya/pingpong-prefilter) (hashing,
  field and point arithmetic, comb, batch inversion, host driver).
- **Prior work:** Jieyi Long's [ecdsafail_gpu_toolkit](https://github.com/jieyilong/ecdsafail_gpu_toolkit) (MIT), the
  first open-source GPU nonce search for this benchmark, whose design this filter shares (no code from it is in this
  repository); the earlier on-GPU finders of John Boyer and runeape-sats; and the early classical prefilters of
  newjordan, alexander-sei and PhantasticUniverse ([`CREDITS.md`](CREDITS.md)).
- **Walk family:** Leapfrog, by BitWonka; the vetoes and choice narrowing the filter models, by haotang1995 and
  odinfree.
- **Baseline circuit:** spookyfrog 24 and the many board entries under it, listed by author in
  [`CREDITS.md`](CREDITS.md).
- **Benchmark:** [ecdsa.fail challenge](https://github.com/Layr-Labs/ecdsafail-challenge) by Layr-Labs.
- **Built with:** Claude Opus 5.5 (xhigh) in Claude Code and GPT-6.1 Sol (xhigh) in Codex CLI.
- **Logo:** generated with Google Gemini.

## Licence

MIT, see [`LICENSE`](LICENSE), except the parts of `src/leapfrog_filter.cu` derived from pingpong-prefilter, which
carries no licence; see [`CREDITS.md`](CREDITS.md).
