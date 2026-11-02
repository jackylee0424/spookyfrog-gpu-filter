//! lftools: stream-binding helpers for the Leapfrog GPU prefilter (CPU only).
//!
//! lftools ckpt <ops.bin> <out checkpoint.bin>
//!     SHAKE256 Fiat-Shamir prefix checkpoint in the PPFSCKP1 layout the filter loads: magic, n_ops, residual_len,
//!     the Keccak state after every full 136-byte block of the stream prefix (everything before the 96-op nonce
//!     tail), the residual bytes, then the 96 tail-op absorb records. Self-checked against the sha3 crate for two
//!     nonces before the file is written.
//! lftools xof <ops.bin> <nonce> <words>
//!     First XOF words of the shot seed for a tail-rewritten nonce (reference sha3 crate), one hex word per line.
//!
//! ops.bin is the harness's "QECCOPSZ" file: a 16-byte header (magic + u64 op count) then one zstd frame of 56-byte
//! records (u32 kind, 4 pad bytes, six u64 fields). The absorbed form of an op is 49 bytes: the kind byte and the six
//! fields, little endian, exactly as the evaluator hashes them.
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::Shake256;
use std::io::{BufReader, Read};

const REC: usize = 56;
const TAIL: usize = 96;
const RATE: usize = 136;
const DOMAIN: &[u8] = b"quantum_ecc-fiat-shamir-v2";

type Absorbed = [u8; 49];

fn absorbed(rec: &[u8; REC]) -> Absorbed {
    assert!(rec[1..8].iter().all(|&b| b == 0), "op kind out of range");
    let mut b = [0u8; 49];
    b[0] = rec[0];
    b[1..].copy_from_slice(&rec[8..]);
    b
}

/// Stream ops.bin: feed every prefix op to `prefix` and return (op count, the 96 tail ops).
fn stream_ops(path: &str, mut prefix: impl FnMut(&Absorbed)) -> (usize, Vec<Absorbed>) {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let mut h = [0u8; 16];
    f.read_exact(&mut h).unwrap();
    assert_eq!(&h[..8], b"QECCOPSZ", "not a QECCOPSZ op stream");
    let n = u64::from_le_bytes(h[8..].try_into().unwrap()) as usize;
    assert!(n > TAIL, "stream shorter than the nonce tail");
    let mut dec = zstd::stream::read::Decoder::new(BufReader::new(f)).unwrap();
    dec.window_log_max(27).unwrap();
    let mut rec = [0u8; REC];
    let mut tail = Vec::with_capacity(TAIL);
    for i in 0..n {
        dec.read_exact(&mut rec).unwrap();
        let a = absorbed(&rec);
        if i < n - TAIL {
            prefix(&a);
        } else {
            tail.push(a);
        }
    }
    (n, tail)
}

fn header(n: usize) -> Vec<u8> {
    let mut v = DOMAIN.to_vec();
    v.extend_from_slice(&(n as u64).to_le_bytes());
    v
}

/// The nonce tail is 48 X;X pairs: op 2b and op 2b+1 target qubit (nonce >> b) & 1.
fn patched(op: &Absorbed, nonce: u64, i: usize) -> Absorbed {
    let mut o = *op;
    o[17..25].copy_from_slice(&((nonce >> (i / 2)) & 1).to_le_bytes());
    o
}

// ---- own Keccak-f[1600], so the absorb state can be checkpointed ----
const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000, 0x000000000000808b,
    0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008a, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000a, 0x000000008000808b, 0x800000000000008b, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const ROT: [u32; 25] = [0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14];

fn keccakf(a: &mut [u64; 25]) {
    for rc in RC {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(ROT[x + 5 * y]);
            }
        }
        for x in 0..5 {
            for y in 0..5 {
                a[x + 5 * y] = b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }
        a[0] ^= rc;
    }
}

#[derive(Clone)]
struct Absorb {
    st: [u64; 25],
    buf: Vec<u8>,
}

impl Absorb {
    fn new() -> Self {
        Absorb { st: [0; 25], buf: Vec::with_capacity(RATE) }
    }
    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.buf.push(b);
            if self.buf.len() == RATE {
                for k in 0..17 {
                    self.st[k] ^= u64::from_le_bytes(self.buf[k * 8..k * 8 + 8].try_into().unwrap());
                }
                keccakf(&mut self.st);
                self.buf.clear();
            }
        }
    }
    /// SHAKE256 padding, then squeeze `n` u64 words.
    fn squeeze_words(&self, n: usize) -> Vec<u64> {
        let mut st = self.st;
        let mut blk = [0u8; RATE];
        blk[..self.buf.len()].copy_from_slice(&self.buf);
        blk[self.buf.len()] ^= 0x1f;
        blk[RATE - 1] ^= 0x80;
        for k in 0..17 {
            st[k] ^= u64::from_le_bytes(blk[k * 8..k * 8 + 8].try_into().unwrap());
        }
        keccakf(&mut st);
        let mut out = Vec::with_capacity(n);
        let mut lane = 0;
        while out.len() < n {
            if lane == 17 {
                keccakf(&mut st);
                lane = 0;
            }
            out.push(st[lane]);
            lane += 1;
        }
        out
    }
}

fn reference_words(prefix: &Shake256, tail: &[Absorbed], nonce: u64, words: usize) -> Vec<u64> {
    let mut h = prefix.clone();
    for (i, op) in tail.iter().enumerate() {
        h.update(&patched(op, nonce, i));
    }
    let mut x = h.finalize_xof();
    let mut out = Vec::with_capacity(words);
    for _ in 0..words {
        let mut b = [0u8; 8];
        XofReader::read(&mut x, &mut b);
        out.push(u64::from_le_bytes(b));
    }
    out
}

fn usage() -> ! {
    eprintln!("usage: lftools ckpt <ops.bin> <out checkpoint.bin>\n       lftools xof <ops.bin> <nonce> <words>");
    std::process::exit(2)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("ckpt") if a.len() == 4 => {
            // the hashed header carries the op count, so read it before streaming the records
            let mut f = std::fs::File::open(&a[2]).unwrap();
            let mut h = [0u8; 16];
            f.read_exact(&mut h).unwrap();
            let n_hdr = u64::from_le_bytes(h[8..].try_into().unwrap()) as usize;
            let mut ab = Absorb::new();
            let mut sh = Shake256::default();
            ab.update(&header(n_hdr));
            sh.update(&header(n_hdr));
            let (n, tail) = stream_ops(&a[2], |op| {
                ab.update(op);
                sh.update(op);
            });
            assert_eq!(n, n_hdr);
            let mut blob = Vec::with_capacity(5064);
            blob.extend_from_slice(b"PPFSCKP1");
            blob.extend_from_slice(&(n as u64).to_le_bytes());
            blob.extend_from_slice(&(ab.buf.len() as u64).to_le_bytes());
            for w in ab.st {
                blob.extend_from_slice(&w.to_le_bytes());
            }
            let mut res = [0u8; RATE];
            res[..ab.buf.len()].copy_from_slice(&ab.buf);
            blob.extend_from_slice(&res);
            for op in &tail {
                // the nonce tail is X gates on qubit 0 or 1
                assert!(op[0] == 6 && op[17] <= 1 && op[18..25].iter().all(|&b| b == 0),
                        "unexpected op in the 96-op nonce tail");
                blob.extend_from_slice(op);
            }
            assert_eq!(blob.len(), 5064);
            for nonce in [0u64, 10002555178u64] {
                let mut ab2 = ab.clone();
                for (i, op) in tail.iter().enumerate() {
                    ab2.update(&patched(op, nonce, i));
                }
                assert_eq!(ab2.squeeze_words(40), reference_words(&sh, &tail, nonce, 40),
                           "checkpoint XOF mismatch for nonce {nonce}");
            }
            std::fs::write(&a[3], &blob).unwrap();
            println!("CKPT ops={n} residual_len={} bytes={} xof_selfcheck=ok", ab.buf.len(), blob.len());
        }
        Some("xof") if a.len() == 5 => {
            let nonce: u64 = a[3].parse().expect("nonce");
            let words: usize = a[4].parse().expect("words");
            let mut f = std::fs::File::open(&a[2]).unwrap();
            let mut h = [0u8; 16];
            f.read_exact(&mut h).unwrap();
            let mut sh = Shake256::default();
            sh.update(&header(u64::from_le_bytes(h[8..].try_into().unwrap()) as usize));
            let (_, tail) = stream_ops(&a[2], |op| sh.update(op));
            for w in reference_words(&sh, &tail, nonce, words) {
                println!("{w:016x}");
            }
        }
        _ => usage(),
    }
}
