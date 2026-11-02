"""Bind the Leapfrog GPU prefilter to a new op stream. See docs/REBIND.md for when each level applies.

    python3 tools/rebind.py check  --checkout DIR [--ops OPS_BIN]   which level a new stream/source needs
    python3 tools/rebind.py level1 --ops OPS_BIN [--build]          new checkpoint + op count (walk and policy unchanged)
    python3 tools/rebind.py level3 --checkout DIR [--build]         regenerate the walk tables from the source
    python3 tools/rebind.py manifest                                 refresh binding.json file hashes after a rebind

Paths are relative to the package (this file's parent directory). DIR is a checkout of the benchmark repository at the
source that produced OPS_BIN (its src/point_add is read, never written). Every rebind invalidates the previous
fidelity audit: rebuild with src/build.sh, then run tools/audit.py before trusting a survivor count.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / 'src'
TOOLS = PKG / 'tools'
MANIFEST = PKG / 'binding.json'
LFTOOLS = TOOLS / 'lftools'
CKP_SIZE = 5064
# files whose bytes decide what the filter computes or how it is bound and checked (doctor compares these)
SHIPPED = ['src/leapfrog_filter.cu', 'src/checkpoint_binding.cuh', 'src/checkpoint.bin', 'src/fe_fast.cuh',
           'src/lf_walk.cuh', 'src/lf_tables.cuh', 'src/fa50_policy.cuh', 'src/sf43h_hint_flat.cuh',
           'src/lf_hint_binding.cuh', 'src/build.sh', 'tools/sfgf.py', 'tools/rebind.py', 'tools/lfmodel.py',
           'tools/nonce.py', 'tools/audit.py', 'tools/validate.py', 'tools/eval_paired.patch', 'tools/make_evaluator.sh',
           'tools/lftools/Cargo.toml', 'tools/lftools/Cargo.lock', 'tools/lftools/src/main.rs']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prefix_sha(ops):
    sys.path.insert(0, str(TOOLS))
    import nonce
    return nonce.prefix_sha256(ops)


def load_manifest():
    return json.loads(MANIFEST.read_text())


def save_manifest(m):
    write_text(MANIFEST, json.dumps(m, indent=2) + '\n')


def refresh_files(m):
    m['files'] = {name: sha(PKG / name) for name in SHIPPED if (PKG / name).exists()}


def init(v):
    return '{' + ','.join(init(x) for x in v) + '}' if isinstance(v, (list, tuple)) else str(v)


def build_lftools():
    exe = LFTOOLS / 'target/release/lftools'
    if not exe.exists():
        subprocess.run(['cargo', 'build', '--release', '--manifest-path', str(LFTOOLS / 'Cargo.toml'),
                        '--target-dir', str(LFTOOLS / 'target')], check=True)
    return exe


def build_filter():
    subprocess.run(['bash', str(SRC / 'build.sh')], check=True)
    return sha(SRC / 'leapfrog_filter')


def write_text(path, text):
    with open(path, 'w', newline='\n') as f:  # LF on every platform: binding.json hashes exact bytes
        f.write(text)


# ---------------------------------------------------------------- level 1: checkpoint + op count
def level1(ops, build):
    ops = Path(ops).resolve()
    head = ops.read_bytes()[:16]
    if head[:8] != b'QECCOPSZ':
        raise SystemExit(f'{ops}: not a QECCOPSZ op stream')
    n_ops = struct.unpack_from('<Q', head, 8)[0]
    tmp = SRC / 'checkpoint.bin.tmp'
    subprocess.run([str(build_lftools()), 'ckpt', str(ops), str(tmp)], check=True)
    ck = tmp.read_bytes()
    assert len(ck) == CKP_SIZE and ck[:8] == b'PPFSCKP1'
    tmp.replace(SRC / 'checkpoint.bin')
    write_text(SRC / 'checkpoint_binding.cuh',
               'static const unsigned char EXPECTED_CHECKPOINT[5064]={' + ','.join(map(str, ck)) + '};\n')
    cu = SRC / 'leapfrog_filter.cu'
    text, count = re.subn(r'#define LF_EXPECTED_OPS \d+ULL', f'#define LF_EXPECTED_OPS {n_ops}ULL', cu.read_text())
    assert count == 1
    write_text(cu, text)
    m = load_manifest()
    m['stream'] = dict(ops=n_ops, ops_sha256=sha(ops), prefix_raw_sha256=prefix_sha(ops),
                       checkpoint_sha256=sha(SRC / 'checkpoint.bin'),
                       bound_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                       label='rebound with tools/rebind.py level1')
    m['audit'] = dict(status='REQUIRED', note='No fidelity audit has been run for this binding. Run tools/audit.py.')
    if build:
        m['build']['binary_sha256'] = build_filter()
        m['build']['arch'] = 'see src/build.sh (ARCH)'
    else:
        m['build']['binary_sha256'] = None
    refresh_files(m)
    save_manifest(m)
    print('LEVEL1_BOUND', json.dumps(dict(ops=n_ops, checkpoint_sha256=m['stream']['checkpoint_sha256'],
                                          binary_sha256=m['build']['binary_sha256'])))
    print('next: ' + ('' if build else 'bash src/build.sh; ') + 'python3 tools/audit.py fidelity ... (docs/REBIND.md, step 4)')


# ---------------------------------------------------------------- level 3: walk tables
def walk_inputs(checkout, m):
    pa = Path(checkout).resolve() / 'src/point_add'
    w = m['walk']
    recipe = (pa / w['recipe']).read_text()
    for needle in w.get('recipe_requires', []):
        if needle not in recipe:
            raise SystemExit(f'{w["recipe"]} lacks {needle!r}: the walk family changed; see docs/REBIND.md, level 3')
    div = int(re.search(r'\("LF_REORDER_DIV", "(\d+)"\)', recipe).group(1))
    mul = int(re.search(r'\("LF_REORDER_MUL", "(\d+)"\)', recipe).group(1))
    late = re.search(r'\("LF_YP8_LATE", "([0-9:,]+)"\)', recipe).group(1)
    schedule = [tuple(map(int, x.split(':'))) for x in late.split(',')]
    rows = lambda p: [list(map(int, l.split())) for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith('#')]
    steps = rows(pa / w['steps'])
    anchors = [int(x) for x in (pa / w['anchors']).read_text().split(',') if x.strip()]
    R = len(steps)
    assert all(len(r) == 5 for r in steps) and len(anchors) >= R and R <= 256
    return pa, dict(div=div, mul=mul, schedule=schedule, late=late, steps=steps, anchors=anchors, R=R)


def level3(checkout, build):
    m = load_manifest()
    pa, w = walk_inputs(checkout, m)
    R, steps, anchors, div, mul = w['R'], w['steps'], w['anchors'], w['div'], w['mul']
    kt = []
    for tick in range(R):
        k = 20
        for start, length in w['schedule']:
            if tick >= start:
                k = length - 2
        kt.append(k)
    spec = importlib.util.spec_from_file_location('lfmodel', TOOLS / 'lfmodel.py')
    lfmodel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lfmodel)
    binding = lfmodel.Binding(steps, anchors, k=20, reorder=(div, mul), m=2)
    eff = [list(part) + [[0] * 6] * (256 - len(part)) for part in binding.eff]
    nl = [(max(*row, steps[min(t + 1, R - 1)][0]) + 2 + 63) // 64 for t, row in enumerate(steps)]
    assert all(1 <= v <= 5 for v in nl)
    masks = [[((1 << wd) - 1) >> (64 * limb) & ((1 << 64) - 1) for limb in range(5)] for wd in range(321)]
    write_text(SRC / 'lf_tables.cuh', f'#define LF_R {R}\n__constant__ int LF_STEPS[{R}][5]=' + init(steps)
               + f';\n__constant__ int LF_ANCHORS[{R}]=' + init(anchors[:R]) + f';\n__constant__ unsigned char LF_KT[{R}]='
               + init(kt) + f';\n\n#define LF_RDIV {div}\n#define LF_RMUL {mul}\n')
    header = ('// Generated by tools/rebind.py level3.\nstatic const unsigned short H_LFEFF[2][256][6]=' + init(eff)
              + ';\nstatic const unsigned short H_LFANCH[256]=' + init(anchors[:R] + [0] * (256 - R))
              + ';\nstatic const unsigned char H_LFNL[256]=' + init(nl + [0] * (256 - R))
              + ';\nstatic const u64 H_LFMASK[321][5]=' + init(masks) + ';\n')
    cut = header.index('static const u64')
    header = header[:cut] + re.sub(r'\b\d+\b', lambda x: x[0] + 'ULL' if int(x[0]) > 2 ** 31 else x[0], header[cut:])
    write_text(SRC / 'lf_hint_binding.cuh', header)
    # the fast selector carries its own compile-time copy of the tables and a 6-tick group schedule
    p = SRC / 'sf43h_hint_flat.cuh'
    s = p.read_text()
    for name, value in [('SF43_ST[][5]', steps), ('SF43_AN[]', anchors[:R]), ('SF43_KT[]', kt), ('SF43_NL[]', nl)]:
        s, c = re.subn(r'^static constexpr int ' + re.escape(name) + r'=.*;$',
                       lambda _: f'static constexpr int {name}=' + init(value) + ';', s, flags=re.M)
        assert c == 1, name
    groups = ''.join(f' if(!sf43h_group<LEG,{lo},{min(lo + 6, R)}>(a,b,seedv,record))return LF_FLAG;\n'
                     for lo in range(0, R, 6))
    s, c = re.subn(r'(?:^ if\(!sf43h_group<LEG,\d+,\d+>\(a,b,seedv,record\)\)return LF_FLAG;\n)+', lambda _: groups, s, flags=re.M)
    assert c == 1
    write_text(p, s)
    m['walk'].update(ticks=R, reorder=[div, mul], yp8_late=w['late'],
                     sources_sha256={n: sha(pa / n) for n in m['walk']['sources']},
                     tables_sha256={n: sha(pa / n) for n in (m['walk']['steps'], m['walk']['anchors'])})
    m['audit'] = dict(status='REQUIRED', note='Walk tables regenerated (level 3). The choice policy (level 2) and the '
                                              'checkpoint (level 1) must match the same stream; then run tools/audit.py.')
    if build:
        m['build']['binary_sha256'] = build_filter()
    refresh_files(m)
    save_manifest(m)
    print('LEVEL3_TABLES', json.dumps(dict(ticks=R, reorder=[div, mul], yp8_late=w['late'])))
    if R != m['policy']['ticks']:
        print(f'WARNING: the choice policy in src/fa50_policy.cuh is bound to {m["policy"]["ticks"]} ticks; this walk has {R}. '
              'Level 2 is mandatory before building.')


# ---------------------------------------------------------------- check
def check(checkout, ops):
    m = load_manifest()
    pa = Path(checkout).resolve() / 'src/point_add'
    w, pol = m['walk'], m['policy']
    out = dict(walk_changed=[], policy_inputs_changed=[], other_changed=[], stream=None)
    for name, want in {**w['sources_sha256'], **w['tables_sha256']}.items():
        if not (pa / name).exists() or sha(pa / name) != want:
            out['walk_changed'].append(name)
    for name, want in pol['inputs_sha256'].items():
        if not (pa / name).exists() or sha(pa / name) != want:
            out['policy_inputs_changed'].append(name)
    known = set(w['sources_sha256']) | set(w['tables_sha256']) | set(pol['inputs_sha256'])
    for name, want in m['source_tree_sha256'].items():
        if name in known:
            continue
        if not (pa / name).exists() or sha(pa / name) != want:
            out['other_changed'].append(name)
    have = {str(p.relative_to(pa)).replace('\\', '/') for p in pa.rglob('*') if p.is_file()}
    out['other_changed'] += sorted(have - set(m['source_tree_sha256']))
    if ops:
        p = prefix_sha(ops)
        out['stream'] = dict(prefix_raw_sha256=p, bound=p == m['stream']['prefix_raw_sha256'])
    if out['walk_changed']:
        level = 3
        why = 'walk sources or tables differ: level 3, then level 2, then level 1'
    elif out['policy_inputs_changed']:
        level = 2
        why = 'choice/hold inputs differ: level 2 (policy), then level 1'
    elif out['other_changed'] or (out['stream'] and not out['stream']['bound']):
        level = 1
        why = ('other src/point_add files (or the stream prefix) differ: level 1. The prehold stream may still have '
               'moved; the fidelity audit decides (any native-good/GPU-rejected shot means level 2 is needed)')
    else:
        level = 0
        why = ('source identical to the bound tree' + (' and stream prefix identical' if ops else
               ' (pass --ops to confirm the stream prefix)') + ': no rebind needed')
    out.update(level=level, why=why)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['check', 'level1', 'level3', 'manifest'])
    ap.add_argument('--ops')
    ap.add_argument('--checkout')
    ap.add_argument('--build', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'check':
        if not a.checkout:
            ap.error('check needs --checkout')
        check(a.checkout, a.ops)
    elif a.cmd == 'level1':
        if not a.ops:
            ap.error('level1 needs --ops')
        level1(a.ops, a.build)
    elif a.cmd == 'level3':
        if not a.checkout:
            ap.error('level3 needs --checkout')
        level3(a.checkout, a.build)
    else:
        m = load_manifest()
        refresh_files(m)
        save_manifest(m)
        print('MANIFEST', len(m['files']), 'files')


if __name__ == '__main__':
    sys.exit(main())
