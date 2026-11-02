"""Exact classical reference of the 86221ad Leapfrog rails walk as the circuit implements it (CPU, pure Python).

Bound to src/point_add/leapfrog.rs (LF_FAST m=2, sign2 + W1, LF_SEED=half, LF_TRIM, LF_SIGNWIRE, LF_LOWREL, LF_REORDER).
Semantics (see GPU-FILTER.md section 1):
  * registers are two's complement; a rail held at width w must satisfy -2^(w-1) <= v < 2^(w-1).  A violation is a
    wrap (add-halve result) or a dirty trim (`resize` CX+R of a non-sign top wire): the rail register then differs
    from the integer the payload tracks, so the shot's classical output is wrong.  Sum overflow of the add before the
    halving is NOT a failure (only the halved value is held).
  * per tick t (target T = R[t%2], source B = R[1-t%2]), effective width checks:
        T  <= n0,  T1 <= min(n0-1, n1),  T2 <= min(n1-1, n2),  T3 <= min(n2-1, nb),  T4 <= W(t+1) (W(t) at the last
        tick; plus W(t) where the tick is reversed rails-only and W is not monotone),  B <= min(n0, n1, n2, W(t+1)).
  * choice: sign2 + W1 window compare [win(|T|') > 4 win(|B|')] on bit positions [anchor-K, anchor+2) of the ones'
    complement magnitude (positions at/above a register's sign wire read 0).
  * endpoint: divide (leg 0) returns y = sign(R0) * P0, so the classical output is right iff R0 = +-1 (R1 != +-1 only
    leaves the P1 register dirty: a reset with a random phase, not a classical error).  Multiply (leg 1) seeds the
    walk-back from the two sign wires, so it needs both rails = +-1.
"""
from pathlib import Path
import json

P = (1 << 256) - (1 << 32) - 977


def fits(v, w):
    return -(1 << (w - 1)) <= v < (1 << (w - 1))


def mag1(v):
    return v if v >= 0 else ~v


class Binding:
    def __init__(self, steps, anchors, k=24, up=2, reorder=(76, 76), m=2):
        self.steps = steps
        self.anchors = anchors
        self.k = k
        self.up = up
        self.m = m
        self.reorder = reorder  # (divide, multiply)
        R = len(steps)
        self.R = R
        W = [s[0] for s in steps]
        mono = all(W[t] >= W[t + 1] for t in range(R - 1))
        self.monotone = mono
        # effective widths per leg per tick: [T, T1, T2, T3, T4, B]
        self.eff = []
        for leg in (0, 1):
            rows = []
            for t, (w, n0, n1, n2, nb) in enumerate(steps):
                wn = W[t + 1] if t + 1 < R else w
                t4 = wn
                # rails-only reverse of tick t starts with resize(tr, W(t)): divide reverses every tick rails-only,
                # multiply the ticks below its reorder point
                rails_only_rev = (leg == 0) or (t < reorder[1])
                if rails_only_rev:
                    t4 = min(t4, w)
                rows.append((n0, min(n0 - 1, n1), min(n1 - 1, n2), min(n2 - 1, nb), t4, min(n0, n1, n2, wn)))
            self.eff.append(rows)

    @classmethod
    def from_source(cls, src_dir, steps_name='env_w1_peel139_steps.txt', anch_name='anch_w1.txt', k=24, reorder=(76, 76)):
        d = Path(src_dir) / 'src' / 'point_add' / 'leapfrog_data'
        steps = [list(map(int, l.split())) for l in (d / steps_name).read_text().splitlines()
                 if l.strip() and not l.startswith('#')]
        anchors = [int(x) for x in (d / anch_name).read_text().split(',') if x.strip()]
        return cls(steps, anchors, k=k, reorder=reorder)

    @classmethod
    def from_json(cls, path):
        b = json.loads(Path(path).read_text())
        return cls(b['steps'], b['anchors'], k=b['w1k'], up=b.get('w1_up', 2),
                   reorder=(b['reorder_div'], b['reorder_mul']), m=b['m'])


def seed(d):
    r0 = d if d & 1 else d - P
    s = r0 + P if (r0 >> 1) & 1 else r0 - P
    return [r0, s >> 1]


def w1(x, b, anchor, k, up):
    h = max(anchor - k, 0)
    l = k + up
    msk = (1 << l) - 1
    return ((mag1(x) >> h) & msk) > 4 * ((mag1(b) >> h) & msk)


def tz(v):
    return (v & -v).bit_length() - 1


def walk(d, bd, leg, trace=False):
    """Returns dict(first=(tick, slot) or None, rails, park, cls) where cls in {'ok', 'fail', 'phase'}."""
    assert bd.m == 2
    r = seed(d)
    first = None
    recs = [] if trace else None
    for t in range(bd.R):
        ti, si = t % 2, 1 - t % 2
        T, B = r[ti], r[si]
        e0, e1, e2, e3, e4, eb = bd.eff[leg][t]
        bad = []
        if not fits(T, e0): bad.append('T')
        if not fits(B, eb): bad.append('B')
        x = T
        S0 = x + B if ((x >> 1) & 1) == ((B >> 1) & 1) else x - B
        x = S0 >> 1
        if not fits(x, e1): bad.append('T1')
        S1 = x + B if ((x >> 1) & 1) == ((B >> 1) & 1) else x - B
        x = S1 >> 1
        if not fits(x, e2): bad.append('T2')
        dq = ((x >> 1) & 1) != ((B >> 1) & 1)
        Dd, F = (x + B, x - B) if dq else (x - B, x + B)
        v = tz(Dd) if Dd != 0 else 999
        if v >= 5:
            deep = False
        elif v >= 3:
            deep = True
        else:
            eq = (x < 0) ^ (B < 0) ^ dq
            deep = (not eq) or w1(x, B, bd.anchors[min(t, len(bd.anchors) - 1)], bd.k, bd.up)
        S3 = Dd if deep else F
        T3 = S3 >> 1
        if not fits(T3, e3): bad.append('T3')
        e = v - 1 if deep else 0
        T4 = T3 >> e
        if not fits(T4, e4): bad.append('T4')
        if bad and first is None:
            first = (t, bad[0])
        r[ti] = T4
        if trace:
            recs.append(dict(t=t, T=T, B=B, T4=T4, e=e, deep=deep, bad=bad))
    if first is not None:
        cls = 'fail'
    elif leg == 0:
        if abs(r[0]) != 1:
            cls = 'fail'
        elif abs(r[1]) != 1:
            cls = 'phase'
        else:
            cls = 'ok'
    else:
        cls = 'ok' if abs(r[0]) == 1 and abs(r[1]) == 1 else 'fail'
    out = dict(first=first, rails=r, cls=cls)
    if trace:
        out['recs'] = recs
    return out


def shot_denominators(tx, ty, ox, oy):
    """Divide walks d1 = tx - ox; multiply walks d2 = ox - ex (mod p)."""
    d1 = (tx - ox) % P
    lam = (ty - oy) * pow(d1, -1, P) % P
    ex = (lam * lam - tx - ox) % P
    d2 = (ox - ex) % P
    return d1, d2
