# Design

## What the benchmark does with a nonce

The trusted evaluator hashes the whole op stream with SHAKE256: the domain string `quantum_ecc-fiat-shamir-v2`, the op
count, then each op as 49 bytes (kind byte and six little-endian u64 fields). It squeezes 9,024 pairs of 256-bit
scalars `(k1, k2)`, forms `T = k1*G` and `O = k2*G` on secp256k1, skips a draw when `T.x == O.x` or either point is
at infinity, and checks that the circuit computes `T + O` on every shot (classical output), leaves no phase garbage
and returns every ancilla to zero. The score is qubits x the rounded mean executed Toffoli count.

The circuit's last 96 ops are 48 `X;X` pairs. Pair `b` targets qubit `(nonce >> b) & 1`, so the pair is the identity
either way, but the hash, and therefore every shot, changes with the nonce. Searching nonces is searching for a draw
where all 9,024 shots pass and the mean Toffoli count rounds low.

## Pipeline per nonce

1. **Fiat-Shamir producer.** The CPU absorbs the nonce-independent prefix once (`tools/lftools ckpt`): the Keccak
   state after every full 136-byte block, the residual bytes and the 96 tail records make up `checkpoint.bin`. The GPU
   patches the 96 tail records for each nonce, finishes the absorb and squeezes the shot scalars, a few thousand Keccak
   permutations per nonce instead of millions.
2. **Points.** `k*G` uses a fixed-base comb (`COMB_BITS`-bit windows; the table is built on the GPU). Jacobian results
   are made affine with a warp-wide Montgomery batch inversion (one field inversion per 128 values). The evaluator's
   skip rule is applied exactly.
3. **Walk inputs.** A second batch inversion gives `d1 = T.x - O.x`, the slope `lambda`, the sum's x and
   `d2 = O.x - E.x`. The circuit's divide leg walks `d1`; its multiply leg walks `d2`.
4. **Tier 1 (screen).** A fast walk (`src/sf43h_hint_flat.cuh`) runs the 122-tick Leapfrog rails on each leg with
   compile-time limb counts per tick (groups of 6 ticks), checks every intermediate against the register widths the
   circuit allocates (`SF43_ST`, the `W n0 n1 n2 nb` table), and checks the endpoint (the divide needs rail 0 = +-1; the
   multiply needs both rails = +-1). It never rejects on its own: a shot it cannot clear is queued.
5. **Tier 3 (confirm).** Queued shots go to the finite-register model (`src/lf_walk.cuh`): two's-complement rails at
   the source's register widths, implicit sign extension, barrel sign fill, the payload modulo p, the walk-back from the
   tape and the unseed check. It decides whether the leg's classical output is wrong. A case outside the model's
   contract rejects (a disclosed yield loss, never a silent admit of a modelled failure).
6. **Verdict.** In screen mode a nonce survives only if no shot of the 9,024 is flagged. Survivors are written as
   little-endian u64 nonces to `--survivor-file`. In exact mode (`MODEL_FLAGS_FILE=path`, no `--screen`) one byte per
   shot is written (1 = modelled classical failure); the audit uses this mode.

## The choices inside the walk

Each tick halves one rail by adding or subtracting the other, possibly twice, then chooses between two moves from a
few low bits and, at YP8 pauses, from a compare of two windows of the rails (`a > 2b` in the plain walk). The window
position comes from the anchor table and the window width from the per-tick `K` schedule (`LF_KT`). On this lineage
the circuit's compare chains are narrowed row by row, so the model uses the exact per-leg, per-tick Boolean function
of the circuit (`src/fa50_policy.cuh`, see `REBIND.md`, level 2). Two boundary rules are lineage specific as well:
the tick-0 seed window is 4 bits wide (from bit 252 of rail 1) and the last tick has no veto window.

## What is not modelled

- **Phase garbage.** `R` and `Hmr` operations kick the global phase; compare windows that are measured out leave phase
  that depends on the shot. The filter cannot see it. In the original search 11 of 423 survivors were classically
  clean (the rest fail in payload cells the model does not carry) and 1 of those 11 was also phase-clean.
- **Approximate payload cells** and **ancilla cleanup** outside the walk.
- **Cost.** The filter does not compute the executed Toffoli count; the stock evaluator does.

So the filter can only reject. Every survivor must go through the stock evaluator, and a found nonce must be rebuilt
into the source and replayed with the stock harness before anyone treats it as a result.

## Safety properties the audit checks

- **No false rejection on fixed packets.** For 48 consecutive nonces the per-shot flags are compared with the stock
  evaluator's wrong shots (a local copy patched to print one `FAILSHOT` line per wrong shot). Any shot flagged by the
  filter but computed correctly by the evaluator fails the audit. Over-admissions (wrong shots the filter misses) are
  expected and caught by validation.
- **Checker control.** The audit injects one false rejection into the flags and requires the comparison to see it.
- **Binding identity.** The binary refuses a checkpoint that differs from the one compiled in, and an op count that
  differs from `LF_EXPECTED_OPS`.

## Command line of the compiled filter

```text
leapfrog_filter --checkpoint src/checkpoint.bin --ops N --from A --to B
                [--batch 65536] [--window 9024] [--chunk 64] [--screen] [--stream] [--async]
                [--survivor-file PATH] [--device K] [--verbose]
```

`tools/sfgf.py scan` uses `--batch 65536 --verbose --screen --stream --async --chunk 64` over windows of 1,000,000
nonces. Nonces must stay below 2^48. `--selftest`, `--reference-fixture` and `--allow-op-mismatch` are disabled in this
build. Environment: `MODEL_FLAGS_FILE` (exact-mode per-shot flags), `LF_INPUTS_FILE` (dump the shot inputs),
`CLASSIC_BLOCKS_PER_SM` (occupancy of the classical kernel).

## Performance

On one RTX 5090 (`ARCH=sm_120`, `MINBLOCKS=3`, `COMB_BITS=16`): about 109,000 nonces/s, about 0.23 survivors per
million nonces on the spookyfrog 24 stream. Build knobs are documented in `src/build.sh`.
