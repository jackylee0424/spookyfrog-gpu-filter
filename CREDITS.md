# Credits

Board entries are cited by the first eight characters of their submission ID, and each ID links to the commit that
the benchmark repository ([Layr-Labs/ecdsafail-challenge](https://github.com/Layr-Labs/ecdsafail-challenge)) made for that entry.
A short hash marked "commit" links to that commit. Submission 98495351 was accepted but never promoted, so it has no
commit.

## Authors

Built by the spookyfrog team for the ecdsa.fail benchmark, with coding agents:

- Claude Opus 5.5 (xhigh), in Claude Code
- GPT-6.1 Sol (xhigh), in Codex CLI

## Logo

`assets/logo.jpg` was generated with Google Gemini.

## Code in this repository

- **Scanner core: John Boyer (johnbxx on the board).** The SHAKE256 prefix checkpoint and nonce-tail patch, the secp256k1 field arithmetic,
  the Jacobian point code, the fixed-base comb, the warp-wide batch inversion and the host driver of
  `src/leapfrog_filter.cu` derive from John Boyer's classical "ping-pong" tail-nonce prefilter, a Rust CPU reference
  (`pingpong_filter.rs`) and a CUDA port, published at https://github.com/somejohnbforya/pingpong-prefilter (commit
  [`da0e572`][pingpong-prefilter da0e572], 2026-08-22). The `pingpong_filter.rs:N` comments in `src/leapfrog_filter.cu` cite the lines of that
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
found in the public submission notes of the board, in order:

- **PhantasticUniverse**: a width and convergence prefilter in front of the full simulator ([b29ff7ff], 2026-06-04).
- **John Boyer (johnbxx)**: a classical convergence prefilter feeding a GPU compare and width screen ([64f75ba9],
  2026-06-05), then an on-GPU finder that hashes, multiplies by the fixed base with a comb and screens every shot
  ([7d1dfe64], 2026-06-06): the earliest GPU nonce screens we found. Later, the ping-pong prefilter that this
  repository's scanner core comes from ([54ff37e6] and [ed30a235], 2026-08-22).
- **alexander-sei**: a classical convergence filter ([488afae5], 2026-06-05), which later notes credit.
- **newjordan**: classical prefilter lanes on rented CPU machines that found clean nonces for new op streams ([012fba1f]
  and following entries, 2026-06-06).
- **runeape-sats**: a CUDA finder of the same shape ([d3b4e7b6] and [2e797801], 2026-06-06).
- **Jieyi Long (jieyilong)**: a bit-exact CUDA port of a circuit's own classical filter, one block per nonce on two
  A100s (98495351, 2026-06-06), then the first open-source GPU nonce search for this benchmark,
  `ecdsafail_gpu_toolkit` (next section).

### Jieyi Long's ecdsafail_gpu_toolkit

https://github.com/jieyilong/ecdsafail_gpu_toolkit, MIT licence, first commit 2026-06-07, announced in submission
[ff46db24] the same day. We read it at commit [`2fdcec6`][ecdsafail_gpu_toolkit 2fdcec6] (2026-06-23).

- **What it is.** A GPU nonce search for the circuit families of June 2026 (the dialog-GCD circuits, later TrailMix). A
  Rust exporter writes the SHAKE256 prefix state, the filter's schedule arrays and a comb table. A CUDA kernel gives
  each nonce one block, whose threads split the 9,024 shots, derives every shot and runs the circuit's own classical
  filter. Survivors go to the stock evaluator. Later commits add batch inversion, larger comb tables, a precomputed
  fan of nonce tails, checks of exact circuit conditions on the CPU, and a skill file for coding agents.
- **Same design, separate code.** This filter uses the same pipeline: a prefix checkpoint so that only the 96-op tail is
  hashed per nonce, a fixed-base comb, batch inversion, a known clean nonce that must survive before any search, and the
  stock evaluator as the final check. We compared the sources line by line. Our CUDA files share no code with the
  toolkit's beyond one `#include` line. The six lines our tools share with its Rust tools and patches are the
  benchmark's own evaluator and Fiat-Shamir code. Our scanner core comes from John Boyer's code (above), and the walk
  model is ours.
- **Its place in this lineage.** krishparadox ran their filter inside the toolkit's kernel ([d7df7706] and [8f980ccb]; the
  second is in the baseline's lineage below), and odinfree used it as their GPU search ([c20c89b3]).
- **Other circuit families.** The toolkit carries filters for the June 2026 families, which this repository does not
  model; this repository models only the Leapfrog walks.

## What the walk model follows

The filter reproduces, on classical values, the walks that the baseline's own source builds. These ideas are theirs:

- **Leapfrog**, the divide and multiply walk family: **BitWonka** (submission 53986ea4, commit [86221ad]).
- **The four-bit tick-0 equal-sign veto window (KH4)**: **haotang1995** ([01a5cc24]).
- **The choice-compare narrowing and the last-tick constant veto**: **haotang1995** ([dca38d60]); the deepened
  choice-narrowing rows at the most deeply narrowed ticks: **odinfree** ([98460d66]). `src/fa50_policy.cuh` encodes the
  rows that this lineage emits.
- **The deeper walk and the tick-0 carry identity**: **zuiris** ([4cb9cf01]); its composition with the spookyfrog square
  and the record-slot, k3, reverse-tick and fold-ladder cuts: **gnuchev** ([88911269], [427ac4c9]).

## The baseline circuit and its lineage

The baseline, spookyfrog 24 (submission [859d928d]), adds one exact table to **odinfree**'s [98460d66]. Its tree carries
the work of, by board entry:

- **odinfree**: [98460d66], [f20f4d87] (a sibling branch, not carried), [57f98ac5] (`cmp_narrow`), [7d81445b] and [db0c5ec7]
  (lane R);
- **haotang1995**: [dca38d60] (tapers, cuts, caps, per-block widths, choice narrowing, vetoes, `cmp_extra`, top-AND cuts,
  `hold_pass`) and [01a5cc24] (KH4);
- **osipovartem**: the row certificates of research submission e813f355 (commit [2d8e5f14], not promoted);
- **Th0rgal (Maxwell on Orb)**: [9ef3139d], the swap-hold table with relays;
- **Champollion9012**: the census tables of [46ae05c5] and [076f1100];
- **mochimodev (Matt Zweil and MAI-A.A.E., Mochimo Foundation)**: [a362dcaa], [c1b95540], [ea64e5cb], [8a1da769] and
  [fa27526a], the tree under all the tables; and Skywalk and U4;
- **gnuchev**: [88911269] and [427ac4c9];
- **zuiris**: [4cb9cf01];
- **Meganpark980320**: [fe0ced81], the swap-park hold form;
- **MonitSharma**: [d7f69382] and [d9ae2845], the AND cache and the general classical passes;
- **kaankolcu**: [767d8d5a], [7927f4b0] and [4daa9b1e], the narrowed compares and census cuts;
- **jackylee0424**: [9da9a45d], [a9b17310], [e4075d6c] and earlier entries, the rail-top tables, the zero-fill rail frame and
  the cache generator;
- **krishparadox**: [8f980ccb], [d4481580] and earlier entries;
- **BitWonka**: Leapfrog;
- **Craig Gidney**: measurement-based uncomputation and deferred phase fixes ([arXiv 2505.15917], section 2.4);
- **the spookyfrog team**: spookyfrog 19 to 24 ([be1f667d], [da097c38], [ee656d09], [45adc970], [859d928d]).

The baseline's public submission note (`ecdsafail submission-note 859d928d`) gives each contribution in full. It also
carries, verbatim, the limiting contracts of mochimodev's note for submission 4f0d2135 (commit [8ee337c]) and of
[PR 419], and states that the
baseline's inherited failure rate (about 21.2 failing shots per 9,024) is above both limits.

## Measurements

Every number in this repository was measured on one RTX 5090 (`sm_120`) under WSL2 unless stated otherwise. See
`VERIFY.json` and `docs/HISTORY.md`.

## Corrections

If your work is used here and is missing or described wrongly, please open an issue.

[b29ff7ff]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/c3c3f8df6664f2560858a5db891450026fd58716
[64f75ba9]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/247db263bc9ceadbef9a16587d80ed937abbffe2
[7d1dfe64]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/8a47cf5d4d3faa52c6ad5cf3490989883a7bc36d
[54ff37e6]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/d6310171d81955af3ff0e196e79a9bd9df95ee0f
[ed30a235]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/787eaa82f257aaee6c16bc08575ddfcfbbedd666
[488afae5]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/fc27501586c327fbc3598a2366955af5c0f13c5f
[012fba1f]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/2f1f1756ba86c740d2c898bda08aaab8de2c6a10
[d3b4e7b6]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/639f12cbe109f087041a7f49269578e53f3951d9
[2e797801]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/571e67ceaf715ed8de2706ece1b4adf6b3a2f851
[ff46db24]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/ea8ce7be09c3784da55dd74278c55598bccba47e
[d7df7706]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/ac4330435196dc54f8000a42835e55fb56f438ae
[8f980ccb]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/cb2df946b406c66fcced5824beb6dba3d61d871e
[c20c89b3]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/7689eacf73722c307606c3decfeec614856ff11c
[01a5cc24]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/b0380209fa7852aee6a5f324dda4000d3dea40ab
[dca38d60]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/e5f4dddc1cadfa427d5ad05cc950cf6038c2e357
[98460d66]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/5ba2d2d4e6fcee9f2f5786d02f26414dafaa53e3
[4cb9cf01]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/f0c0a662dc6e104905cf235991959bb758783f87
[88911269]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/a14a87d287160b1165525bd0391250c95e4f4413
[427ac4c9]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/72ccdb80efc31e020525736c90c0b3aae33d226e
[859d928d]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/39b7cd573f5a7a6b01970a9cd4f11a462e398ea1
[f20f4d87]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/9160fded8ce5458e37eb72f1a5d1ca6eed82254e
[57f98ac5]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/e4076aebde2695f09ef32764f7749968b6d39763
[7d81445b]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/3519e5bca81cd0242524ea9a16aeda7eb9364e96
[db0c5ec7]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/b50b24f9d39f726d15f3cd26ab560b38c59958f2
[9ef3139d]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/3a0a1cb254088f1dd6b456b992a70ec5c3be93fa
[46ae05c5]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/6b0327d60e40068fcbed5fedcb7f66d926f12320
[076f1100]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/9599459c030d35bda7d2043d52817aa50b0b9940
[a362dcaa]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/572967a1e785ca15299453505d585af56a3e8b4e
[c1b95540]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/86d8a6582e4f6c7f7fb4c57a340342b3198656d7
[ea64e5cb]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/16beb9769862e9b9c116961695a0b145b8029a89
[8a1da769]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/917c33d1f0d2d1430d48cfbd4ca836f72889b2f7
[fa27526a]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/457beedbd2714e2faf08f51725e773a8b5458ca3
[fe0ced81]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/7ed0f830787b60c3ad17a80337ab3858968a5bc3
[d7f69382]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/0ae6d109bee6f3241e1f74c38fddb5a05b7f79de
[d9ae2845]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/90e777ea8242faa0acb2462c5cb5be7c318a2d64
[767d8d5a]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/441d9ef9f5fe05107f4390e9f623caffdf77349d
[7927f4b0]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/143731efbffe897e8df00af9c276a4121b8ae253
[4daa9b1e]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/ca00b2886c4f6715f6f3e8bd33244c5bc4839aec
[9da9a45d]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/a62b71da0061246110edd07e5d6db4333a199ae3
[a9b17310]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/31c38da873433027b99c346e1ebef82aa6de07b7
[e4075d6c]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/6602a4e835898b9a4118f040806195571c6e8967
[d4481580]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/c672dda072a9601ed9359c869752d30f6e583c94
[be1f667d]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/97698072b8dd5d2c830248cfe87ff8eb2ec83e7d
[da097c38]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/644857a7724560048a4ce23b3d8364a81010881c
[ee656d09]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/ba8ad25654a74a0910f539b920a2fcd350e53098
[45adc970]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/2120d8a20c9d277e3cfdae29639ec58a78e72e16
[86221ad]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/86221adc33c4621d9d595894ab1b2683172f11c0
[8ee337c]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/8ee337cb69f7f56d4f04b03a951bf538479f2ee0
[2d8e5f14]: https://github.com/Layr-Labs/ecdsafail-challenge/commit/2d8e5f140930e1610c4e1d4406169e9fd58c4a05
[pingpong-prefilter da0e572]: https://github.com/somejohnbforya/pingpong-prefilter/commit/da0e5721f14f5d956aa703b38311f4087388a7b1
[ecdsafail_gpu_toolkit 2fdcec6]: https://github.com/jieyilong/ecdsafail_gpu_toolkit/commit/2fdcec6e755b34cdf7c11b9e36e9abf03701e6ec
[PR 419]: https://github.com/Layr-Labs/ecdsafail-challenge/pull/419
[arXiv 2505.15917]: https://arxiv.org/abs/2505.15917
