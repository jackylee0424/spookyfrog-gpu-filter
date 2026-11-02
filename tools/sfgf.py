#!/usr/bin/env python3
"""spookyfrog-gpu-filter driver: one command per replication step (AGENTS.md gives the order).

    python3 tools/sfgf.py doctor                       what is installed, what is missing
    python3 tools/sfgf.py fetch-baseline               benchmark checkout at the baseline commit -> work/baseline
    python3 tools/sfgf.py build-stream                 stock harness build + evaluation -> work/baseline/ops.bin
    python3 tools/sfgf.py check                        which rebind level the stream needs (0 = none)
    python3 tools/sfgf.py rebind --level 1|3           bind the filter to work/baseline (level 2 is manual: docs/REBIND.md)
    python3 tools/sfgf.py build [--arch sm_XX]         compile src/leapfrog_filter
    python3 tools/sfgf.py audit [--start N]            fidelity audit gate (must PASS before any scan)
    python3 tools/sfgf.py control                      positive control + replay of the known clean nonce
    python3 tools/sfgf.py scan --from N --count M      screen nonces on the GPU (refuses an unaudited binary)
    python3 tools/sfgf.py validate (--nonce N | --survivors FILE)   full stock-evaluator check

Each command ends with one line `SFGF <command> <json>` and exits 0 on PASS, 1 on FAIL, 2 on a setup or usage error.
Defaults come from expected/spookyfrog24.json (override with --expected). Work files go to ./work (git-ignored).
"""
import argparse
import ctypes.util
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC, TOOLS = REPO / 'src', REPO / 'tools'
WORK = REPO / 'work'
sys.path.insert(0, str(TOOLS))
import nonce as nonce_tool  # noqa: E402

SCREEN = ['--batch', '65536', '--verbose', '--screen', '--stream', '--async', '--chunk', '64']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def done(cmd, ok, **fields):
    fields = dict(ok=bool(ok), **fields)
    print(f'SFGF {cmd} ' + json.dumps(fields), flush=True)
    return 0 if ok else 1


def fail_setup(cmd, msg, **fields):
    print(f'SFGF {cmd} ' + json.dumps(dict(ok=False, error=msg, **fields)), flush=True)
    return 2


def run(cmd, log=None, **kw):
    """Run a command; tee stdout+stderr to `log` when given. Returns CompletedProcess with text output."""
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if log:
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f'$ {" ".join(map(str, cmd))}\n{p.stdout}\n{p.stderr}')
    return p


def last_json(text, tag):
    for line in reversed(text.splitlines()):
        if line.startswith(tag + ' '):
            return json.loads(line[len(tag) + 1:])
    return None


def expected(a):
    return json.loads(Path(a.expected).read_text())


def baseline_dir(a):
    return Path(a.dir).resolve() if getattr(a, 'dir', None) else (WORK / 'baseline')


def c_compiler():
    for c in [os.environ.get('CC'), 'gcc', 'cc', 'clang']:
        if c and shutil.which(c):
            return shutil.which(c)
    return None


def cargo_env():
    """The same RUSTFLAGS benchmark.sh uses, so the evaluator build reuses its compiled dependencies."""
    cc = c_compiler()
    env = dict(os.environ)
    if cc:
        env['RUSTFLAGS'] = f'-C linker={cc}'
        env['CC'] = cc
    return env


def nvcc():
    return shutil.which('nvcc') or ('/usr/local/cuda/bin/nvcc' if Path('/usr/local/cuda/bin/nvcc').exists() else None)


def gpu_arch():
    if not shutil.which('nvidia-smi'):
        return None, None
    p = run(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'])
    if p.returncode or not p.stdout.strip():
        return None, None
    name, cap = [x.strip() for x in p.stdout.strip().splitlines()[0].split(',')]
    return name, 'sm_' + cap.replace('.', '')


# ---------------------------------------------------------------- doctor
def doctor(a):
    checks = []

    def add(name, ok, required, detail, hint=None):
        checks.append(dict(name=name, ok=bool(ok), required=required, detail=detail, **({'hint': hint} if hint and not ok else {})))

    add('linux', sys.platform.startswith('linux'), True, platform.platform(), 'Use Linux or WSL2.')
    add('python>=3.9', sys.version_info >= (3, 9), True, sys.version.split()[0])
    n = nvcc()
    nv = run([n, '--version']).stdout if n else ''
    rel = re.search(r'release ([0-9.]+)', nv)
    add('nvcc', bool(n), True, rel.group(1) if rel else None, 'Install the CUDA toolkit (12.8+ for sm_120) and put nvcc on PATH.')
    name, arch = gpu_arch()
    add('nvidia_gpu', bool(name), True, dict(name=name, arch=arch), 'nvidia-smi must list a GPU (WSL2: install the Windows NVIDIA driver).')
    for tool, hint in [('cargo', 'Install Rust (rustup); the benchmark harness is Rust.'), ('rustc', 'Install Rust (rustup).'),
                       ('git', 'Install git.'), ('patch', 'Install patch.'), ('tar', 'Install tar.')]:
        add(tool, shutil.which(tool), True, shutil.which(tool), hint)
    add('c_compiler', c_compiler(), True, c_compiler(), 'Install gcc (the harness links with it).')
    add('libzstd', ctypes.util.find_library('zstd'), True, ctypes.util.find_library('zstd'), 'Install libzstd1 (apt install zstd).')
    add('bubblewrap', shutil.which('bwrap'), False, shutil.which('bwrap'),
        'Optional: the harness sandboxes build_circuit with bwrap and otherwise runs it unconfined.')
    cli = shutil.which('ecdsafail') or (str(Path.home() / '.local/bin/ecdsafail') if (Path.home() / '.local/bin/ecdsafail').exists() else None)
    add('ecdsafail_cli', cli, False, cli, 'Optional: only needed to submit; not needed to replicate.')
    m = json.loads((REPO / 'binding.json').read_text())
    bad = [f for f, h in m.get('files', {}).items() if not (REPO / f).exists() or sha(REPO / f) != h]
    add('repo_files_match_binding', not bad, True, bad or 'all files match binding.json',
        'Files changed since the last rebind: run `python3 tools/rebind.py manifest` if you rebound on purpose, else re-clone.')
    WORK.mkdir(exist_ok=True)
    free = shutil.disk_usage(WORK).free / 2 ** 30
    add('disk_free_gb>=6', free >= 6, True, round(free, 1), 'The baseline checkout and its build need about 4 GB.')
    b = WORK / 'baseline'
    state = dict(baseline=(b / '.git').exists(), stream=(b / 'ops.bin').exists(),
                 filter_built=(SRC / 'leapfrog_filter').exists(),
                 evaluator_built=(b / 'target/release/eval_paired').exists())
    missing = [c['name'] for c in checks if c['required'] and not c['ok']]
    for c in checks:
        print(('ok   ' if c['ok'] else ('MISS ' if c['required'] else 'opt  ')) + f"{c['name']}: {c['detail']}"
              + (f"  -> {c['hint']}" if 'hint' in c else ''))
    return done('doctor', not missing, missing=missing, suggested_arch=arch, state=state, checks=checks)


# ---------------------------------------------------------------- baseline
def fetch_baseline(a):
    e = expected(a)
    d = baseline_dir(a)
    commit = a.commit or e['baseline']['commit']
    if (d / '.git').exists():
        head = run(['git', '-C', str(d), 'rev-parse', 'HEAD']).stdout.strip()
        return done('fetch-baseline', head == commit, dir=str(d), head=head, expected=commit,
                    note='existing checkout reused' if head == commit else
                    'existing checkout is at another commit: remove it or pass --dir')
    d.mkdir(parents=True)
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    if a.from_checkout:
        steps = [['git', 'clone', '-q', '--no-checkout', str(Path(a.from_checkout).resolve()), str(d)]]
    else:
        steps = [['git', 'init', '-q', str(d)],
                 ['git', '-C', str(d), 'remote', 'add', 'origin', a.url or e['baseline']['repo_url']],
                 ['git', '-C', str(d), 'fetch', '-q', '--depth', '1', 'origin', commit]]
    for step in steps:
        p = run(step, env=env)
        if p.returncode:
            return fail_setup('fetch-baseline', 'git failed: ' + ' '.join(step[:4]), stderr=p.stderr[-800:])
    p = run(['git', '-C', str(d), 'checkout', '-q', commit if a.from_checkout else 'FETCH_HEAD'])
    head = run(['git', '-C', str(d), 'rev-parse', 'HEAD']).stdout.strip()
    tail = re.search(r'"TAIL_NONCE" => "(\d+)"', (d / 'src/point_add/mod.rs').read_text())
    return done('fetch-baseline', head == commit, dir=str(d), head=head, expected=commit,
                baked_nonce=int(tail.group(1)) if tail else None)


def build_stream(a):
    e = expected(a)
    d = baseline_dir(a)
    if not (d / 'benchmark.sh').exists():
        return fail_setup('build-stream', f'{d} is not a benchmark checkout: run fetch-baseline')
    logs = WORK / 'logs'
    t0 = time.monotonic()
    p = run(['cargo', 'fetch', '--locked'], log=logs / 'cargo-fetch.log', cwd=d)
    if p.returncode:
        return fail_setup('build-stream', 'cargo fetch failed (network?)', log='work/logs/cargo-fetch.log')
    p = run(['bash', './benchmark.sh', '--note', 'sfgf build-stream'], log=logs / 'build-stream.log', cwd=d)
    ops = d / 'ops.bin'
    if not ops.exists():
        return fail_setup('build-stream', 'benchmark.sh produced no ops.bin', log='work/logs/build-stream.log')
    n_ops = struct.unpack_from('<Q', ops.read_bytes()[:16], 8)[0]
    score = json.loads((d / 'score.json').read_text()) if (d / 'score.json').exists() else None
    s = e['stream']
    got = dict(ops=n_ops, prefix_raw_sha256=nonce_tool.prefix_sha256(ops), raw_sha256=nonce_tool.raw_sha256(ops),
               score=score and score.get('score'), harness_rc=p.returncode)
    match = dict(ops=got['ops'] == s['ops'], prefix_raw_sha256=got['prefix_raw_sha256'] == s['prefix_raw_sha256'],
                 raw_sha256=got['raw_sha256'] == s['raw_sha256'], score=got['score'] == e['baseline']['score'])
    return done('build-stream', all(match.values()), **got, match=match, seconds=round(time.monotonic() - t0),
                log='work/logs/build-stream.log')


# ---------------------------------------------------------------- binding
def check(a):
    d = baseline_dir(a)
    ops = Path(a.ops).resolve() if a.ops else d / 'ops.bin'
    p = run([sys.executable, str(TOOLS / 'rebind.py'), 'check', '--checkout', str(d)] + (['--ops', str(ops)] if ops.exists() else []))
    if p.returncode:
        return fail_setup('check', 'rebind.py check failed', stderr=p.stderr[-800:])
    r = json.loads(p.stdout)
    return done('check', True, level=r['level'], why=r['why'], walk_changed=r['walk_changed'],
                policy_inputs_changed=r['policy_inputs_changed'], other_changed=r['other_changed'][:20], stream=r['stream'])


def rebind(a):
    d = baseline_dir(a)
    if a.level == 2:
        print('Level 2 (choice policy) is a manual procedure: docs/REBIND.md, "Level 2".')
        return fail_setup('rebind', 'level 2 is manual (docs/REBIND.md)')
    if a.level == 1:
        ops = Path(a.ops).resolve() if a.ops else d / 'ops.bin'
        p = run([sys.executable, str(TOOLS / 'rebind.py'), 'level1', '--ops', str(ops)], log=WORK / 'logs/rebind-level1.log')
    else:
        p = run([sys.executable, str(TOOLS / 'rebind.py'), 'level3', '--checkout', str(d)], log=WORK / 'logs/rebind-level3.log')
    print(p.stdout, end='')
    if p.returncode:
        return fail_setup('rebind', 'rebind failed', stderr=p.stderr[-800:])
    m = json.loads((REPO / 'binding.json').read_text())
    e = expected(a)
    return done('rebind', True, level=a.level, checkpoint_sha256=m['stream']['checkpoint_sha256'],
                matches_shipped_baseline_checkpoint=m['stream']['checkpoint_sha256'] == e['stream']['checkpoint_sha256'],
                next='python3 tools/sfgf.py build, then audit')


def build(a):
    _, arch = gpu_arch()
    arch = a.arch or os.environ.get('ARCH') or arch or 'sm_120'
    p = run(['bash', str(SRC / 'build.sh')], log=WORK / 'logs/build-filter.log', env=dict(os.environ, ARCH=arch))
    if p.returncode or not (SRC / 'leapfrog_filter').exists():
        return fail_setup('build', 'nvcc build failed', log='work/logs/build-filter.log', stderr=p.stderr[-800:])
    h = sha(SRC / 'leapfrog_filter')
    (WORK / 'build.json').write_text(json.dumps(dict(arch=arch, binary_sha256=h, checkpoint_sha256=sha(SRC / 'checkpoint.bin'),
                                                     utc=now()), indent=1) + '\n')
    return done('build', True, arch=arch, binary_sha256=h)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------- gates
def ensure_evaluator(d):
    exe = d / 'target/release/eval_paired'
    if exe.exists():
        return exe
    env = cargo_env()
    env['CARGO_FLAGS'] = '--locked --offline'
    p = run(['bash', str(TOOLS / 'make_evaluator.sh'), str(d)], log=WORK / 'logs/make-evaluator.log', env=env)
    if p.returncode or not exe.exists():
        raise RuntimeError('evaluator build failed: see work/logs/make-evaluator.log')
    return exe


def audit(a):
    e = expected(a)
    d = baseline_dir(a)
    ops = Path(a.ops).resolve() if a.ops else d / 'ops.bin'
    if not (SRC / 'leapfrog_filter').exists():
        return fail_setup('audit', 'filter not built: python3 tools/sfgf.py build')
    try:
        evaluator = ensure_evaluator(d)
    except RuntimeError as x:
        return fail_setup('audit', str(x))
    start = a.start if a.start is not None else e['audit']['start']
    out = WORK / 'audits' / f'fidelity-{start}-{a.count}-{sha(SRC / "leapfrog_filter")[:12]}'
    if (out / 'AUDIT.json').exists():
        r = json.loads((out / 'AUDIT.json').read_text())
        return done('audit', r['passed'], reused=str(out.relative_to(REPO)), **r['summary'], packets=r['packets'])
    if out.exists():
        shutil.rmtree(out)
    p = run([sys.executable, str(TOOLS / 'audit.py'), 'fidelity', '--ops', str(ops), '--evaluator', str(evaluator),
             '--start', str(start), '--count', str(a.count), '--out', str(out), '--jobs', str(a.jobs)],
            log=WORK / 'logs' / f'audit-{start}.log')
    r = last_json(p.stdout, 'AUDIT')
    if not r:
        return fail_setup('audit', 'audit did not finish', log=f'work/logs/audit-{start}.log', stderr=p.stderr[-800:])
    return done('audit', r['passed'], audit=str((out / 'AUDIT.json').relative_to(REPO)), **{k: v for k, v in r.items() if k != 'passed'})


def audited_binary():
    h = sha(SRC / 'leapfrog_filter')
    for f in sorted((WORK / 'audits').glob('*/AUDIT.json')):
        r = json.loads(f.read_text())
        if r.get('passed') and r.get('binary_sha256') == h and r.get('packets', 0) >= 48:
            return f
    return None


def control(a):
    e = expected(a)
    d = baseline_dir(a)
    ops = d / 'ops.bin'
    n = e['control']['nonce']
    p = run([sys.executable, str(TOOLS / 'audit.py'), 'positive', '--nonce', str(n)], log=WORK / 'logs/control-positive.log')
    pos = last_json(p.stdout, 'POSITIVE')
    if not pos:
        return fail_setup('control', 'positive screen failed', stderr=p.stderr[-800:])
    stock = d / 'target/release/eval_circuit'
    if not stock.exists():
        return fail_setup('control', 'stock evaluator missing: run build-stream')
    p = run([sys.executable, str(TOOLS / 'validate.py'), 'check', '--ops', str(ops), '--evaluator', str(stock), '--nonce', str(n)],
            log=WORK / 'logs/control-validate.log')
    v = last_json(p.stdout, 'VALIDATE')
    if not v:
        return fail_setup('control', 'validation failed to run', stderr=p.stderr[-800:])
    raw = nonce_tool.raw_sha256(ops, n)
    score = (v.get('score_json') or {}).get('score')
    checks = dict(survives_filter=pos['survived'], clean=v['clean'], score=score == e['control']['score'],
                  replay_raw=raw == e['control']['raw_sha256'])
    return done('control', all(checks.values()), nonce=n, checks=checks, score=score, mean_toffoli=v['mean_toffoli'],
                qubits=v['qubits'], replication='PASS' if all(checks.values()) else 'FAIL')


def scan(a):
    m = json.loads((REPO / 'binding.json').read_text())
    exe = SRC / 'leapfrog_filter'
    if not exe.exists():
        return fail_setup('scan', 'filter not built')
    ok_audit = audited_binary()
    if not ok_audit and not a.unaudited:
        return fail_setup('scan', 'this binary has no passing fidelity audit (>= 48 packets): run `sfgf.py audit` first')
    out = Path(a.out).resolve() if a.out else WORK / 'scans' / str(a.lo)
    (out / 'windows').mkdir(parents=True, exist_ok=True)
    lo, end, survivors, rates = a.lo, a.lo + a.count, [], []
    while lo < end:
        hi = min(lo + a.window, end)
        surv_file = out / 'windows' / f'{lo}.bin'
        jf = out / 'windows' / f'{lo}.json'
        if jf.exists():
            w = json.loads(jf.read_text())
        else:
            surv_file.unlink(missing_ok=True)
            t0 = time.monotonic()
            p = run([str(exe), '--checkpoint', str(SRC / 'checkpoint.bin'), '--ops', str(m['stream']['ops']),
                     '--from', str(lo), '--to', str(hi - 1)] + SCREEN + ['--survivor-file', str(surv_file)])
            if p.returncode:
                return fail_setup('scan', 'filter failed', stderr=p.stderr[-800:])
            data = surv_file.read_bytes() if surv_file.exists() else b''
            ns = sorted(int.from_bytes(data[i:i + 8], 'little') for i in range(0, len(data), 8))
            r = re.findall(r'= ([0-9.]+) nonces/s', p.stderr + p.stdout)
            w = dict(start=lo, end_exclusive=hi, survivors=len(ns), nonces=ns,
                     nonce_s=float(r[-1]) if r else round((hi - lo) / (time.monotonic() - t0)))
            jf.write_text(json.dumps(w, indent=1) + '\n')
        survivors += w['nonces']
        rates.append(w['nonce_s'])
        lo = hi
    allf = out / 'survivors.bin'
    allf.write_bytes(b''.join(int(x).to_bytes(8, 'little') for x in sorted(set(survivors))))
    # windows that coincide with the original search must reproduce its survivor sets exactly
    ref = expected(a).get('reference_windows', {})
    compared, mismatched = 0, []
    if m['stream']['prefix_raw_sha256'] == expected(a)['stream']['prefix_raw_sha256'] and a.window == ref.get('window'):
        for start, want in ref.get('survivors', {}).items():
            jf = out / 'windows' / f'{start}.json'
            if jf.exists():
                compared += 1
                if sorted(json.loads(jf.read_text())['nonces']) != sorted(want):
                    mismatched.append(int(start))
    return done('scan', not mismatched, start=a.lo, nonces=a.count, survivors=len(set(survivors)),
                nonces_per_second=round(sum(rates) / len(rates)) if rates else None,
                reference_windows_compared=compared, reference_windows_mismatched=mismatched,
                survivors_file=str(allf), audited_by=str(ok_audit) if ok_audit else 'UNAUDITED',
                next='python3 tools/sfgf.py validate --survivors ' + str(allf))


def validate(a):
    d = baseline_dir(a)
    ops = Path(a.ops).resolve() if a.ops else d / 'ops.bin'
    stock = d / 'target/release/eval_circuit'
    if not stock.exists():
        return fail_setup('validate', 'stock evaluator missing: run build-stream')
    cmd = [sys.executable, str(TOOLS / 'validate.py'), 'check', '--ops', str(ops), '--evaluator', str(stock), '--jobs', str(a.jobs)]
    cmd += ['--nonce', str(a.nonce)] if a.nonce is not None else ['--survivors', str(a.survivors)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    rows = [json.loads(l[len('VALIDATE '):]) for l in p.stdout.splitlines() if l.startswith('VALIDATE ')]
    for r in rows:
        print('VALIDATE', json.dumps(r))
    clean = [dict(nonce=r['nonce'], mean_toffoli=r['mean_toffoli'], score=(r.get('score_json') or {}).get('score')) for r in rows if r['clean']]
    return done('validate', p.returncode == 0, checked=len(rows), clean=clean)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--expected', default=str(REPO / 'expected/spookyfrog24.json'))
    ap.add_argument('--dir', help='benchmark checkout (default work/baseline)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('doctor')
    f = sub.add_parser('fetch-baseline')
    f.add_argument('--commit')
    f.add_argument('--url')
    f.add_argument('--from-checkout', help='clone from a local benchmark clone instead of the network')
    sub.add_parser('build-stream')
    c = sub.add_parser('check')
    c.add_argument('--ops')
    r = sub.add_parser('rebind')
    r.add_argument('--level', type=int, choices=[1, 2, 3], required=True)
    r.add_argument('--ops')
    b = sub.add_parser('build')
    b.add_argument('--arch')
    u = sub.add_parser('audit')
    u.add_argument('--start', type=int)
    u.add_argument('--count', type=int, default=48)
    u.add_argument('--jobs', type=int, default=4)
    u.add_argument('--ops')
    sub.add_parser('control')
    s = sub.add_parser('scan')
    s.add_argument('--from', dest='lo', type=int, required=True)
    s.add_argument('--count', type=int, required=True)
    s.add_argument('--window', type=int, default=1000000)
    s.add_argument('--out')
    s.add_argument('--unaudited', action='store_true', help='allow a binary without a passing audit (not recommended)')
    v = sub.add_parser('validate')
    g = v.add_mutually_exclusive_group(required=True)
    g.add_argument('--nonce', type=int)
    g.add_argument('--survivors')
    v.add_argument('--ops')
    v.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    WORK.mkdir(exist_ok=True)
    fn = {'doctor': doctor, 'fetch-baseline': fetch_baseline, 'build-stream': build_stream, 'check': check, 'rebind': rebind,
          'build': build, 'audit': audit, 'control': control, 'scan': scan, 'validate': validate}[a.cmd]
    return fn(a)


if __name__ == '__main__':
    sys.exit(main())
