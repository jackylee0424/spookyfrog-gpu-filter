"""Op-stream helpers: rewrite the 96-op nonce tail of an ops.bin, or decode it to raw records (Linux, libzstd).

    python3 tools/nonce.py rewrite OPS_BIN NONCE OUT_BIN     write OPS_BIN with its nonce tail set to NONCE
    python3 tools/nonce.py rawsha OPS_BIN [NONCE]            SHA-256 of the decoded records (after a tail rewrite)
    python3 tools/nonce.py prefixsha OPS_BIN                 SHA-256 of the decoded records before the nonce tail

ops.bin ("QECCOPSZ") is a 16-byte header (magic + u64 op count) and one zstd frame of 56-byte records. The tail is 48
X;X pairs on qubit 0 or 1: op n-96+2b and op n-96+2b+1 target qubit (nonce >> b) & 1, the same rule the circuit's
source applies when it bakes a nonce in. The rewrite never touches the prefix (asserted by hash). Compare decoded
records, not compressed files: two encoders can frame the same stream differently.
"""
import ctypes
import ctypes.util
import hashlib
import json
import struct
import sys
from pathlib import Path

REC = 56
TAIL = 96


def _zstd():
    name = ctypes.util.find_library('zstd')
    if not name:
        raise SystemExit('libzstd not found (install libzstd1 / zstd)')
    z = ctypes.CDLL(name)
    for fn, args in [('ZSTD_decompress', [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]),
                     ('ZSTD_compress', [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]),
                     ('ZSTD_compressBound', [ctypes.c_size_t]), ('ZSTD_isError', [ctypes.c_size_t])]:
        f = getattr(z, fn)
        f.argtypes = args
        f.restype = ctypes.c_size_t
    return z


def load_raw(path):
    """Return (header bytes, op count, decoded record buffer)."""
    blob = Path(path).read_bytes()
    if blob[:8] != b'QECCOPSZ':
        raise SystemExit(f'{path}: not a QECCOPSZ op stream')
    count = struct.unpack_from('<Q', blob, 8)[0]
    assert count > TAIL
    z = _zstd()
    raw = ctypes.create_string_buffer(count * REC)
    data = blob[16:]
    r = z.ZSTD_decompress(raw, len(raw), data, len(data))
    assert not z.ZSTD_isError(r) and r == len(raw), 'zstd frame does not decode to count * 56 bytes'
    return blob[:16], count, raw


def set_tail(raw, count, nonce):
    assert 0 <= nonce < 2 ** 48
    tail = (count - TAIL) * REC
    for i in range(TAIL):
        off = tail + i * REC
        assert struct.unpack_from('<I', raw, off)[0] == 6, 'tail op is not X'
        assert struct.unpack_from('<Q', raw, off + 24)[0] in (0, 1), 'tail X is not on qubit 0 or 1'
        struct.pack_into('<Q', raw, off + 24, (nonce >> (i // 2)) & 1)


def raw_sha256(path, nonce=None):
    _, count, raw = load_raw(path)
    if nonce is not None:
        set_tail(raw, count, nonce)
    return hashlib.sha256(raw).hexdigest()


def prefix_sha256(path):
    """SHA-256 of the decoded records before the nonce tail: the part of a stream a binding depends on."""
    _, count, raw = load_raw(path)
    return hashlib.sha256(memoryview(raw).cast('B')[:(count - TAIL) * REC]).hexdigest()


def rewrite(source, nonce, out):
    head, count, raw = load_raw(source)
    view = memoryview(raw).cast('B')
    tail = (count - TAIL) * REC
    before = hashlib.sha256(view[:tail]).hexdigest()
    set_tail(raw, count, nonce)
    assert hashlib.sha256(view[:tail]).hexdigest() == before
    z = _zstd()
    packed = ctypes.create_string_buffer(z.ZSTD_compressBound(len(raw)))
    size = z.ZSTD_compress(packed, len(packed), raw, len(raw), 3)
    assert not z.ZSTD_isError(size)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(head)
        f.write(memoryview(packed).cast('B')[:size])
    tmp.replace(out)
    return {'nonce': nonce, 'ops': count, 'prefix_raw_sha256': before, 'raw_sha256': hashlib.sha256(raw).hexdigest()}


if __name__ == '__main__':
    a = sys.argv[1:]
    if len(a) == 4 and a[0] == 'rewrite':
        print(json.dumps(rewrite(Path(a[1]), int(a[2]), Path(a[3]))))
    elif len(a) in (2, 3) and a[0] == 'rawsha':
        print(raw_sha256(a[1], int(a[2]) if len(a) == 3 else None))
    elif len(a) == 2 and a[0] == 'prefixsha':
        print(prefix_sha256(a[1]))
    else:
        raise SystemExit(__doc__)
