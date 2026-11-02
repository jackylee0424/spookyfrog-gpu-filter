# History

## Lineage of the filter

| date (2026) | step |
|---|---|
| Jun 4-7 | Classical nonce prefilters appear on the board (PhantasticUniverse, alexander-sei, John Boyer, newjordan), and Jieyi Long publishes the first bit-exact CUDA port and the public `ecdsafail_gpu_toolkit`, whose design this filter follows (`CREDITS.md`). |
| Aug 22 | John Boyer published a classical "ping-pong" tail-nonce prefilter (CPU reference + CUDA port, github.com/somejohnbforya/pingpong-prefilter). We reviewed it and kept its scanner core: the SHAKE256 prefix checkpoint, field arithmetic, comb and batch inversion (`CREDITS.md`). |
| Oct 6 | First Leapfrog filter, for the Leapfrog circuit family (BitWonka, submission 86221ad, 1,243 qubits): classical models of the divide and multiply walks, run-time binding of the walk tables. |
| Oct 9-25 | Rebinds for each new head of the lineage; the A2 rails (122 ticks, replay boundaries 37 and 43) and the fast tier-1 selector (`sf43h_hint_flat.cuh`); full multiply and divide models; queued finite-register confirmation (`lf_walk.cuh`). |
| Nov 2 | The repair described below (`fa50_policy.cuh`), then the search that found spookyfrog 24. |

## The Nov 2 repair: from 79 false rejections per 48 packets to 0 per 192

**Symptom.** Bound to the stream of submission 98460d66 (a public lineage built on our earlier tree that narrows the
comparator chains, `choice_narrow.txt`, and adds hold rows), the filter had passed its known-answer test and ran at
about 120,000 nonces/s. A first search of 2.09 billion nonces found no clean nonce. We stopped it and audited the
binding on fixed packets before searching further.

**Diagnosis.** On 48 fixed packets (433,152 shots) the filter flagged 79 shots that the stock evaluator computes
correctly: 76 at the YP8 choice compare and 3 at the tick-0 seed window. That is 1.65 false rejections per packet,
so the chance that a clean nonce survives was about e^-1.65, roughly 20%: the filter was discarding about 80% of the
clean nonces. A second panel showed the same (74 false rejections).

**Repair (model only; the circuit and the stream did not change).**

- The YP8 choice compare became the exact per-leg, per-tick function of the circuit's narrowed comparator chain
  (476 rows; 238 forward functions extracted from the operations before the hold pass; `fa50_policy.cuh`).
- The tick-0 seed window narrowed from 12 to 4 bits (from bit 252 of rail 1).
- No veto window at the last tick.

**Verification.**

| panel | packets | native-good / GPU-rejected | over-admitted shots |
|---|---|---|---|
| original panel (unrepaired filter) | 48 | 79 | 331 |
| original panel (repaired) | 48 | 0 | 187 |
| fresh panel (repaired) | 48 | 0 | 179 |
| unseen panel A (repaired) | 48 | 0 | 165 |
| unseen panel B (repaired) | 48 | 0 | 177 |
| **repaired total** | **192 (1,732,608 shots)** | **0** | |

The one classically clean survivor of the stopped search still survived the repaired filter. Throughput went from about
120,000 to 109,000 nonces/s.

**Outcome.** A second search over 1,814,000,000 nonces (265 minutes, start 1,452,086,000,000) kept 423 survivors;
11 were classically clean and 1 was clean in every check: nonce `1453896738487`, mean 739,464.902 Toffoli at 1,234
qubits, score 912,499,810 (spookyfrog 24, accepted and promoted). The windows listed in
`expected/spookyfrog24.json` come from that search.

**Lesson.** Audit fixed packets against the stock evaluator after every rebind, not only a known-answer test. The
known-answer test checks the hash and the point arithmetic; only the audit checks that the walk model agrees with the
circuit you bound it to.
