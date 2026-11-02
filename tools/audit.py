"""Qualify a bound prefilter build before trusting it. Run after EVERY rebind (docs/REBIND.md, step 4).

    python3 tools/audit.py fidelity --ops OPS_BIN --evaluator EVAL_PAIRED --start N --count 48 --out DIR [--jobs 4]
    python3 tools/audit.py positive --nonce N [--expect survive|reject]
    python3 tools/audit.py throughput --from N --count 1000000

fidelity    For COUNT consecutive nonces from START (a diagnostic range you will not scan for real), compare the
            filter's per-shot flags (exact mode, 1 = modelled classical walk failure) with the stock evaluator's wrong
            shots on the same nonce (EVAL_PAIRED from tools/make_evaluator.sh, one FAILSHOT line per wrong shot).
            PASS requires native_good_gpu_bad == 0: the filter must never flag a shot the evaluator computes
            correctly, because such a flag rejects a nonce that may be clean. Shots the evaluator gets wrong and the
            filter passes (native_bad_gpu_good) are expected over-admissions; the full check of each survivor catches
            them. A checker control injects one false rejection and requires the reconciliation to see it.
positive    Screen one nonce and report whether it survives (use a nonce you know is clean on this stream).
throughput  Screen a window and report nonces/s and survivors per million.

Every number is written to DIR/AUDIT.json (fidelity) or printed as one JSON line.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / 'src'
TOOLS = PKG / 'tools'
sys.path.insert(0, str(TOOLS))
import nonce as nonce_tool  # noqa: E402

SHOTS = 9024
SCREEN = ['--batch', '65536', '--verbose', '--screen', '--stream', '--async', '--chunk', '64']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding():
    m = json.loads((PKG / 'binding.json').read_text())
    exe = SRC / 'leapfrog_filter'
    if not exe.exists():
        raise SystemExit('src/leapfrog_filter is not built: bash src/build.sh')
    if sha(SRC / 'checkpoint.bin') != m['stream']['checkpoint_sha256']:
        raise SystemExit('src/checkpoint.bin does not match binding.json: rebind (tools/rebind.py) first')
    return m, exe


def base_cmd(m, exe):
    return [str(exe), '--checkpoint', str(SRC / 'checkpoint.bin'), '--ops', str(m['stream']['ops'])]


def screen(m, exe, lo, hi):
    with tempfile.TemporaryDirectory() as td:
        surv = Path(td) / 'surv.bin'
        t0 = time.monotonic()
        p = subprocess.run(base_cmd(m, exe) + ['--from', str(lo), '--to', str(hi)] + SCREEN + ['--survivor-file', str(surv)],
                           capture_output=True, text=True)
        wall = time.monotonic() - t0
        if p.returncode:
            raise SystemExit(p.stderr[-3000:])
        data = surv.read_bytes() if surv.exists() else b''
    survivors = sorted(int.from_bytes(data[i:i + 8], 'little') for i in range(0, len(data), 8))
    rate = re.findall(r'= ([0-9.]+) nonces/s', p.stderr + p.stdout)
    return survivors, (float(rate[-1]) if rate else (hi - lo + 1) / wall), wall


def cpu_wrong(evaluator, ops, n, work):
    d = work / str(n)
    d.mkdir()
    one = d / 'ops.bin'
    nonce_tool.rewrite(ops, n, one)
    p = subprocess.run([str(evaluator)], cwd=d, env=dict(os.environ, EVAL_OPS=str(one)), capture_output=True, text=True)
    text = p.stdout + p.stderr
    one.unlink()
    m = re.search(r'tested shots\s*:\s*(\d+)', text)
    if not m:
        raise SystemExit(f'evaluator produced no shot count for nonce {n}:\n{text[-2000:]}')
    wrong = sorted(int(x) for x in re.findall(r'^FAILSHOT (\d+)', text, re.M))
    count = re.search(r'classical mismatches\s*:\s*(\d+)', text)
    assert count and int(count.group(1)) == len(wrong), f'nonce {n}: FAILSHOT lines disagree with the mismatch count'
    phase = len(re.findall(r'^PHASEBATCH ', text, re.M))
    return n, int(m.group(1)), wrong, phase


def reconcile(flags, start, results):
    tot = dict(native_good_gpu_bad=0, native_bad_gpu_good=0, true_reject=0, true_admit=0)
    rows = []
    for i, (n, tested, wrong, phase) in enumerate(results):
        assert n == start + i and tested == SHOTS
        ff = flags[i * SHOTS:(i + 1) * SHOTS]
        bad = set(wrong)
        fn = [s for s in range(SHOTS) if ff[s] and s not in bad]
        fp = [s for s in wrong if not ff[s]]
        tot['native_good_gpu_bad'] += len(fn)
        tot['native_bad_gpu_good'] += len(fp)
        tot['true_reject'] += sum(1 for s in range(SHOTS) if ff[s]) - len(fn)
        tot['true_admit'] += SHOTS - len(bad) - len(fn)
        rows.append(dict(nonce=n, cpu_wrong=len(wrong), cpu_phase_batches=phase, gpu_flagged=sum(1 for b in ff if b),
                         native_good_gpu_bad=fn, native_bad_gpu_good=len(fp)))
    return tot, rows


def fidelity(a):
    m, exe = binding()
    ops, evaluator, out = Path(a.ops).resolve(), Path(a.evaluator).resolve(), Path(a.out).resolve()
    if nonce_tool.prefix_sha256(ops) != m['stream']['prefix_raw_sha256']:
        raise SystemExit('OPS_BIN is not the bound stream (prefix differs): rebind first')
    if out.exists():
        raise SystemExit(f'{out} exists: use a fresh output directory')
    out.mkdir(parents=True)
    t0 = time.monotonic()
    flags_file = out / 'gpu.flags'
    p = subprocess.run(base_cmd(m, exe) + ['--from', str(a.start), '--to', str(a.start + a.count - 1), '--batch', '65536',
                                           '--verbose'], env=dict(os.environ, MODEL_FLAGS_FILE=str(flags_file)),
                       capture_output=True, text=True)
    (out / 'gpu.stderr').write_text(p.stderr)
    if p.returncode:
        raise SystemExit(p.stderr[-3000:])
    flags = flags_file.read_bytes()
    assert len(flags) == a.count * SHOTS
    work = out / 'cpu'
    work.mkdir()
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda n: cpu_wrong(evaluator, ops, n, work), range(a.start, a.start + a.count)))
    tot, rows = reconcile(flags, a.start, results)
    # checker control: one injected false rejection must be counted
    first_bad = set(results[0][2])
    probe = next(s for s in range(SHOTS) if s not in first_bad and not flags[s])
    injected = bytearray(flags)
    injected[probe] = 1
    ctot, _ = reconcile(bytes(injected), a.start, results)
    control = dict(shot=probe, nonce=a.start, detected=ctot['native_good_gpu_bad'] == tot['native_good_gpu_bad'] + 1)
    assert control['detected'], 'checker control failed: the reconciliation missed an injected false rejection'
    res = dict(passed=tot['native_good_gpu_bad'] == 0, packets=a.count, shots=a.count * SHOTS, start=a.start, summary=tot,
               model_survivors=sum(1 for i in range(a.count) if not any(flags[i * SHOTS:(i + 1) * SHOTS])),
               cpu_classically_clean=sum(1 for r in results if not r[2]), checker_control=control, rows=rows,
               binary_sha256=sha(exe), checkpoint_sha256=m['stream']['checkpoint_sha256'],
               evaluator_sha256=sha(evaluator), flags_sha256=sha(flags_file), seconds=round(time.monotonic() - t0, 1),
               scope='Finite fixed-packet check of the classical walk model only. Phase and payload cells are not '
                     'modelled; every survivor still needs the full evaluator run (tools/validate.py).')
    (out / 'AUDIT.json').write_text(json.dumps(res, indent=1) + '\n')
    print('AUDIT', json.dumps(dict(passed=res['passed'], packets=a.count, **tot, checker_control=control['detected'])))
    return 0 if res['passed'] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fidelity')
    f.add_argument('--ops', required=True)
    f.add_argument('--evaluator', required=True)
    f.add_argument('--start', type=int, required=True)
    f.add_argument('--count', type=int, default=48)
    f.add_argument('--out', required=True)
    f.add_argument('--jobs', type=int, default=4)
    p = sub.add_parser('positive')
    p.add_argument('--nonce', type=int, required=True)
    p.add_argument('--expect', choices=['survive', 'reject'], default='survive')
    t = sub.add_parser('throughput')
    t.add_argument('--from', dest='lo', type=int, required=True)
    t.add_argument('--count', type=int, default=1000000)
    a = ap.parse_args()
    if a.cmd == 'fidelity':
        return fidelity(a)
    m, exe = binding()
    if a.cmd == 'positive':
        surv, _, _ = screen(m, exe, a.nonce, a.nonce)
        ok = (a.nonce in surv) == (a.expect == 'survive')
        print('POSITIVE', json.dumps(dict(nonce=a.nonce, survived=a.nonce in surv, expected=a.expect, ok=ok)))
        return 0 if ok else 1
    surv, rate, wall = screen(m, exe, a.lo, a.lo + a.count - 1)
    print('THROUGHPUT', json.dumps(dict(start=a.lo, nonces=a.count, nonces_per_second=round(rate), seconds=round(wall, 2),
                                        survivors=len(surv), survivors_per_million=round(len(surv) * 1e6 / a.count, 1),
                                        survivor_nonces=surv)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
