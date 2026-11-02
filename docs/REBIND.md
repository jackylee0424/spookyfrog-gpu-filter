# Rebinding the filter to another stream

The filter is compiled for one op stream. Using it on another circuit (a new submission, your own edit, a rival's
tree) means rebinding it, rebuilding it and **auditing it again**. A binding that runs is not a binding that is right:
on one lineage a binding that passed a known-answer test still rejected about 80% of clean nonces
([`HISTORY.md`](HISTORY.md)).

## What a binding consists of

| part | files | derived from |
|---|---|---|
| hash checkpoint | `src/checkpoint.bin`, `src/checkpoint_binding.cuh` | the stream prefix: domain header + op count + every op before the 96-op nonce tail, then the 96 tail records |
| op count | `#define LF_EXPECTED_OPS` in `src/leapfrog_filter.cu` | the stream header |
| walk tables | `src/lf_tables.cuh`, `src/lf_hint_binding.cuh`, the `SF43_*` arrays and the 6-tick group list in `src/sf43h_hint_flat.cuh` | `src/point_add/a2_rails.rs` (`LF_REORDER_DIV`, `LF_REORDER_MUL`, `LF_YP8_LATE`), `leapfrog_data/a2_R122_steps.txt`, `leapfrog_data/a2_R122_anch.txt` |
| choice policy | `src/fa50_policy.cuh`, plus the tick-0 seed window and last-tick rule in `src/lf_walk.cuh` and `src/sf43h_hint_flat.cuh` | the circuit's own comparator operations, read from the stream before the hold pass |

The binary refuses to run when `--ops` or the checkpoint file differs from what it was compiled with. `binding.json`
records the hashes of the sources, tables and choice inputs the shipped binding was made from.

## Which level

```bash
cd PATH_TO_CHECKOUT && ./benchmark.sh          # stock harness: writes ops.bin (and scores the baked nonce)
cd -   # back to this repository
python3 tools/sfgf.py --dir PATH_TO_CHECKOUT check
```

| level | condition (as `check` reports it) | apply |
|---|---|---|
| 0 | `src/point_add` identical to the bound tree and the same stream prefix | nothing; every nonce of this source already works |
| 1 | other `src/point_add` files differ (later tables, builder changes); walk sources, tables and choice inputs identical | level 1 |
| 2 | `choice_narrow.txt`, `choice_narrow.rs` or `hold_pass.rs` differ, **or** a level 1 audit finds native-good/GPU-rejected shots | level 2, then level 1 |
| 3 | `a2_rails.rs`, `y15_a2.rs`, `y15_a2r1.rs`, `leapfrog.rs` or the `leapfrog_data` tables differ | level 3, then level 2, then level 1 |

`check` is a file comparison, so level 1 is necessary but not sufficient: an edit anywhere before the hold pass can
move the operations the choice policy was read from. **The audit decides.** If the circuit no longer uses the A2
rails recipe, or its walk differs in structure (tick logic, letters, more than 256 ticks), this filter does not apply
and the model has to be ported.

## Procedure

1. Build the new stream with its own harness (`./benchmark.sh` in its checkout).
2. `python3 tools/sfgf.py --dir CHECKOUT check` and note the level.
3. Apply the level:
   - **Level 1:** `python3 tools/sfgf.py --dir CHECKOUT rebind --level 1`. This runs `tools/lftools ckpt` (which checks
     its own Keccak checkpoint against the reference SHA-3 implementation for two nonces), rewrites
     `src/checkpoint.bin`, `src/checkpoint_binding.cuh` and `LF_EXPECTED_OPS`, and updates `binding.json`.
   - **Level 3:** `python3 tools/sfgf.py --dir CHECKOUT rebind --level 3`. Regenerates every walk table from the checkout
     (it requires `pub(crate) const A2_DEFAULT: bool = true;` and the three recipe keys). On the shipped tree it
     reproduces the shipped tables byte for byte. Continue with level 2 and level 1.
   - **Level 2:** manual, below.
4. Build and audit:
   ```bash
   python3 tools/sfgf.py --dir CHECKOUT build
   python3 tools/sfgf.py --dir CHECKOUT audit --start S        # S: a range you will never scan
   ```
   Pass: `native_good_gpu_bad == 0` and `checker_control == true`. One panel is 48 packets (433,152 shots). We shipped
   after four panels (192 packets); run at least a second panel at another start when the level was 2 or 3.
5. Positive control. If you know a clean nonce on the new stream: `python3 tools/audit.py positive --nonce N` must say
   `survived: true`. If you do not, scan a few million nonces and look at the survivor rate: on this lineage a healthy
   binding keeps about 0.2 nonces per million. Zero survivors over tens of millions is a false-reject symptom.
6. Only now scan and validate (`AGENTS.md`, steps 8-9).

To bind back to the baseline, rerun level 1 on the baseline `ops.bin`; it reproduces the shipped checkpoint
(`865ab30e...`).

## Level 2: regenerate the choice policy (manual)

**Why it exists.** At each YP8 pause the circuit chooses between two moves by comparing two windows of the rails
(`a > 2b` in the plain walk). Lineages that narrow those comparator chains (the rows in `choice_narrow.txt` drop low
bits, removed carries or the top AND) compute a slightly different Boolean function. The model must use the function
the circuit really computes, per leg and per tick, or it flags shots the circuit gets right.

**Procedure.**

1. *Observer build.* Copy the checkout. In the copy's `src/point_add/mod.rs`, dump the op list immediately before
   `choice_narrow::run(ops)` (call it prechoice) and immediately before `hold_pass::hold_pass(ops)` (prehold), one
   56-byte record per op in the `ops.bin` record layout:
   ```rust
   fn sfgf_dump(ops: &[Op], path: &str) {
       use std::io::Write;
       let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
       for o in ops {
           f.write_all(&(o.kind as u32).to_le_bytes()).unwrap();
           f.write_all(&0u32.to_le_bytes()).unwrap();
           for v in [o.q_control2.0, o.q_control1.0, o.q_target.0, o.c_target.0, o.c_condition.0, o.r_target.0] {
               f.write_all(&v.to_le_bytes()).unwrap();
           }
       }
       f.flush().unwrap();
   }
   ```
   Build `build_circuit` and run it in a scratch directory (`benchmark.sh` deletes its scratch directory, so run the
   binary yourself). Confirm the emitted stream decodes to the same records as the stock one
   (`python3 tools/nonce.py rawsha` on both): observation must not change the circuit.
2. *Inventory.* In the prechoice dump, find the comparator chain units (an `X` on y, `CCX(x, y -> t)`, `X` on y,
   bracketed by the `CX` fan-out from the carry wire c) and match them to the rows of `choice_narrow.txt`
   (key: t, x, y, c, occurrence; value: depth d, top flag, removed count). The shipped lineage has 476 rows:
   119 ticks x 2 directions x 2 legs.
3. *Extract.* Locate each row's chain in the prehold dump and take the operations from the chain start to the compare
   output wire (the wire later XORed into the paused add's not-flipped bit).
4. *Evaluate.* For the 238 forward rows (119 ticks x 2 legs), evaluate the slice symbolically over the window bits
   `a_j`, `b_j`: `X` adds 1, `CX` XORs, `CCX` XORs an AND. The result is an exact XOR-AND function of the window bits.
5. *Emit* `src/fa50_policy.cuh` in the shipped form: `template<int LEG,int T> int fa50_y(u64 a, u64 b)` returning that
   function for leg `LEG` (0 = divide, 1 = multiply) and tick `T` (1..R-3), and `fa50_y_runtime` dispatching on
   `leg*R + t`. Where a row is equivalent to a masked compare `((a&M)>>d) > (((b<<1)&BM)>>d)` on the window widths,
   that form is shorter; the shipped file uses it for 11 rows (each checked against its extracted network) and the
   extracted network itself for the other 227. Emitting the network for every row is always exact.
6. Check the two tick-boundary rules the shipped `lf_walk.cuh` and `sf43h_hint_flat.cuh` carry for this lineage: a
   4-bit tick-0 seed window read from bit 252 of rail 1, and no veto window at the last tick. A lineage that changes
   tick-0 or last-tick logic needs the same treatment.
7. Rebuild, run two audit panels, run the positive control.

## After a rebind

`binding.json` and the regenerated headers change; that is expected. `python3 tools/sfgf.py doctor` checks that the
files still match `binding.json` (the rebind tools update it). Keep the audit output (`work/audits/*/AUDIT.json`) with
any result you report.
