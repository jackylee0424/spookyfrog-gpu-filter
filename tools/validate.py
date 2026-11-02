"""Full check of prefilter survivors with the stock evaluator, and the replay check before any submission.

    python3 tools/validate.py check --ops OPS_BIN --evaluator EVAL_CIRCUIT (--nonce N | --survivors FILE) [--jobs 2]
    python3 tools/validate.py replay --ops OPS_BIN --nonce N --built BUILT_OPS_BIN

check   For each nonce: rewrite the stream's nonce tail (tools/nonce.py), run the STOCK evaluator (the harness's
        eval_circuit, which reads ./ops.bin) in a scratch directory, and report the four checks, the executed
        Toffoli mean and the score (qubits x mean rounded half away from zero, as the evaluator computes it; with a
        single nonce the evaluator's own score.json, written in its checkout root, is read back and compared).
        A nonce is clean only if the evaluator itself says every shot is OK. The prefilter does not see phase or
        payload cells, so most survivors fail here; that is expected.
replay  After you bake a clean nonce into the source and rebuild with the stock harness, compare the decoded records of
        the rebuilt ops.bin with the rewritten stream. They must be identical (compressed files may differ). Then run
        the harness benchmark on that tree before submitting anything.
"""
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import nonce as nonce_tool  # noqa: E402


def field(pattern, text, cast=int):
    m = re.search(pattern, text, re.M)
    return cast(m.group(1)) if m else None


def check_one(ops, evaluator, n, read_score_json):
    # The stock evaluator reads ./ops.bin but writes score.json (and appends results.tsv) in its own checkout root, so
    # score.json is only read back when a single evaluation runs; the score is also recomputed from the printed mean.
    root_score = Path(evaluator).resolve().parents[2] / 'score.json'
    with tempfile.TemporaryDirectory(prefix=f'lf-validate-{n}-') as td:
        d = Path(td)
        nonce_tool.rewrite(ops, n, d / 'ops.bin')
        t0 = time.time()
        p = subprocess.run([str(evaluator)], cwd=d, capture_output=True, text=True)
        text = p.stdout + p.stderr
    score = None
    if read_score_json and root_score.exists() and root_score.stat().st_mtime >= t0 - 1:
        score = json.loads(root_score.read_text())
    r = dict(nonce=n, returncode=p.returncode,
             tested_shots=field(r'tested shots\s*:\s*(\d+)', text),
             classical_mismatches=field(r'classical mismatches\s*:\s*(\d+)', text),
             phase_garbage_batches=field(r'phase-garbage batches\s*:\s*(\d+)', text),
             ancilla_garbage_batches=field(r'ancilla-garbage batches\s*:\s*(\d+)', text),
             all_shots_ok=bool(re.search(r'all \d+ shots OK', text)),
             mean_toffoli=field(r'avg executed Toffoli\s*:\s*([0-9.]+)', text, float),
             qubits=field(r'^\s*qubits\s*:\s*(\d+)', text), score_json=score)
    r['clean'] = (p.returncode == 0 and r['all_shots_ok'] and r['classical_mismatches'] == 0
                  and r['phase_garbage_batches'] == 0 and r['ancilla_garbage_batches'] == 0)
    if r['clean'] and r['mean_toffoli'] is not None and r['qubits']:
        printed = field(r'avg executed Toffoli\s*:\s*([0-9.]+)', text, str)
        r['rounded_toffoli'] = math.floor(r['mean_toffoli'] + 0.5)  # the evaluator rounds half away from zero
        r['score'] = r['rounded_toffoli'] * r['qubits']
        # the mean is printed to 3 decimals; .500 cannot be rounded safely from the printed value
        r['score_ambiguous'] = printed.endswith('.500')
        if score:
            r['score_matches_score_json'] = score.get('score') == r['score']
    if not r['clean']:
        r['reason'] = (re.findall(r'!! .*', text) or ['(see evaluator output)'])[0]
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--ops', required=True)
    c.add_argument('--evaluator', required=True, help='the stock target/release/eval_circuit')
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument('--nonce', type=int)
    g.add_argument('--survivors', help='survivor file written by the filter (little-endian u64 nonces)')
    c.add_argument('--jobs', type=int, default=2)
    r = sub.add_parser('replay')
    r.add_argument('--ops', required=True)
    r.add_argument('--nonce', type=int, required=True)
    r.add_argument('--built', required=True)
    a = ap.parse_args()
    if a.cmd == 'replay':
        want = nonce_tool.raw_sha256(a.ops, a.nonce)
        got = nonce_tool.raw_sha256(a.built)
        print('REPLAY', json.dumps(dict(nonce=a.nonce, rewritten_raw_sha256=want, built_raw_sha256=got, identical=want == got)))
        return 0 if want == got else 1
    ops, evaluator = Path(a.ops).resolve(), Path(a.evaluator).resolve()
    if a.nonce is not None:
        nonces = [a.nonce]
    else:
        data = Path(a.survivors).read_bytes()
        nonces = sorted({int.from_bytes(data[i:i + 8], 'little') for i in range(0, len(data), 8)})
    jobs = max(1, min(a.jobs, len(nonces)))
    with ThreadPoolExecutor(jobs) as pool:
        for res in pool.map(lambda n: check_one(ops, evaluator, n, jobs == 1), nonces):
            print('VALIDATE', json.dumps(res), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
