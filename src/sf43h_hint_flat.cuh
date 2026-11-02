// Leapfrog (86221ad family) walk verdict for one leg of one shot.  Port of tools/lfmodel.py (tier 1) and
// an exact register-level reference (tier 3); see README.md.  Route tables come from the binding at run time.
//
//   tier 1: ideal walk (letters from the true rails; exact while every value fits its register), need-based width
//           checks with the per-leg effective widths [T, T1, T2, T3, T4, B]; any miss, or a wrong endpoint, escalates.
//   tier 3: register-exact rails (two's-complement contents at the source's register widths, implicit sign extensions,
//           barrel sign fill, LF_TRIM/LF_SIGNWIRE payload-tick widths, reorder), letters from contents, payload mod p,
//           endpoint, walk-back from the tape and the unseed x check.  Decides the leg's classical outcome.
#pragma once

#define LF_MAXR 256
#define LF_ST_W 0
__constant__ int c_lfR, c_lfK, c_lfUP, c_lfRDIV, c_lfRMUL, c_lfW0;
__constant__ unsigned short c_lfst[LF_MAXR][5];   // W n0 n1 n2 nb
__constant__ unsigned short c_lfanch[LF_MAXR];
__constant__ unsigned short c_lfeff[2][LF_MAXR][6]; // tier-1 effective widths per leg: T, T1, T2, T3, T4, B
__constant__ u64 c_lfmask[321][5];                // low-w-bit masks, w = 0..320

enum { LF_OK = 0, LF_FLAG = 1, LF_PHASE = 2 };

// ---------------------------------------------------------------- 320-bit two's complement helpers
__device__ __forceinline__ void s5_copy(u64 r[5], const u64 a[5]) {
#pragma unroll
    for (int i = 0; i < 5; ++i) r[i] = a[i];
}
// r = a + (sub ? -b : b)  (mod 2^320)
__device__ __forceinline__ void s5_addsub(u64 r[5], const u64 a[5], const u64 b[5], u32 sub) {
    u64 m = 0ULL - (u64)sub;
    u64 o0 = b[0] ^ m, o1 = b[1] ^ m, o2 = b[2] ^ m, o3 = b[3] ^ m, o4 = b[4] ^ m;
    u64 r0, r1, r2, r3, r4;
    asm("{.reg .u32 c;\n\t"
        "add.cc.u32 c, 0xffffffff, %15;\n\t"
        "addc.cc.u64 %0, %5, %10;\n\t"
        "addc.cc.u64 %1, %6, %11;\n\t"
        "addc.cc.u64 %2, %7, %12;\n\t"
        "addc.cc.u64 %3, %8, %13;\n\t"
        "addc.u64    %4, %9, %14;\n\t}"
        : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3), "=l"(r4)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]), "l"(a[4]),
          "l"(o0), "l"(o1), "l"(o2), "l"(o3), "l"(o4), "r"(sub));
    r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3; r[4] = r4;
}
// arithmetic shift right by 1 <= k <= 63
__device__ __forceinline__ void s5_sar(u64 r[5], const u64 a[5], int k) {
    u64 t[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) t[i] = (a[i] >> k) | (a[i + 1] << (64 - k));
    t[4] = (u64)((long long)a[4] >> k);
    s5_copy(r, t);
}
// logical shift right by 0 <= k <= 63
__device__ __forceinline__ void s5_shr(u64 r[5], const u64 a[5], int k) {
    if (k == 0) { s5_copy(r, a); return; }
    u64 t[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) t[i] = (a[i] >> k) | (a[i + 1] << (64 - k));
    t[4] = a[4] >> k;
    s5_copy(r, t);
}
// logical shift left by 0 <= k <= 63
__device__ __forceinline__ void s5_shl(u64 r[5], const u64 a[5], int k) {
    if (k == 0) { s5_copy(r, a); return; }
    u64 t[5];
    t[0] = a[0] << k;
#pragma unroll
    for (int i = 1; i < 5; ++i) t[i] = (a[i] << k) | (a[i - 1] >> (64 - k));
    s5_copy(r, t);
}
__device__ __forceinline__ u64 s5_bit(const u64 a[5], int i) {
    int wd = i >> 6, sh = i & 63;
    u64 x = 0;
#pragma unroll
    for (int k = 0; k < 5; ++k) x = (k == wd) ? a[k] : x;
    return (x >> sh) & 1ULL;
}
// signed value fits in e bits: bits >= e-1 all equal the sign
__device__ __forceinline__ bool s5_fits(const u64 a[5], int e) {
    u64 fill = (u64)((long long)a[4] >> 63), acc = 0;
    const u64 *m = c_lfmask[e - 1];
#pragma unroll
    for (int j = 0; j < 5; ++j) acc |= (a[j] ^ fill) & ~m[j];
    return acc == 0;
}
__device__ __forceinline__ bool s5_is_pm1(const u64 a[5]) {
    bool one = a[0] == 1ULL && (a[1] | a[2] | a[3] | a[4]) == 0;
    bool neg = (a[0] & a[1] & a[2] & a[3] & a[4]) == ~0ULL;
    return one || neg;
}
// bits [h, h+n) of v ^ (sgmask) as an integer (n <= 32), v a 5-limb word vector
__device__ __forceinline__ u64 s5_window(const u64 v[5], u64 sgmask, int h, int n) {
    if (n <= 0) return 0;
    int wd = h >> 6, sh = h & 63;
    u64 lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 5; ++k) { lo = (k == wd) ? v[k] : lo; hi = (k == wd + 1) ? v[k] : hi; }
    lo ^= sgmask;
    hi = (wd + 1 < 5) ? (hi ^ sgmask) : 0ULL;
    u64 x = (lo >> sh) | (sh ? (hi << (64 - sh)) : 0ULL);
    return x & ((1ULL << n) - 1ULL);
}

// ---------------------------------------------------------------- seeds (LF_SEED=half)
// R0 = d (odd) or d - p; R1 = (R0 + (R0 == 3 mod 4 ? p : -p)) / 2
__device__ __forceinline__ void lf_seed(u64 r0[5], u64 r1[5], const u64 d[4]) {
    u64 dd[5] = { d[0], d[1], d[2], d[3], 0 };
    u64 p5[5] = { c_PMOD[0], c_PMOD[1], c_PMOD[2], c_PMOD[3], 0 };
    if (d[0] & 1) s5_copy(r0, dd); else s5_addsub(r0, dd, p5, 1);
    u64 t[5];
    s5_addsub(t, r0, p5, ((r0[0] >> 1) & 1) ? 0u : 1u);
    s5_sar(r1, t, 1);
}

// ---------------------------------------------------------------- tier 1
__device__ int lf_tier1(const u64 d[4], int leg) {
    u64 r[2][5];
    lf_seed(r[0], r[1], d);
    const int R = c_lfR;
    for (int t = 0; t < R; ++t) {
        const int ti = t & 1;
        u64 T[5], B[5];
        s5_copy(T, r[ti]); s5_copy(B, r[ti ^ 1]);
        const unsigned short *ew = c_lfeff[leg][t];
        if (!s5_fits(T, ew[0]) || !s5_fits(B, ew[5])) return LF_FLAG;
        u32 s0 = (u32)(((T[0] ^ B[0]) >> 1) & 1);
        s5_addsub(T, T, B, s0); s5_sar(T, T, 1);
        if (!s5_fits(T, ew[1])) return LF_FLAG;
        u32 s1 = (u32)(((T[0] ^ B[0]) >> 1) & 1);
        s5_addsub(T, T, B, s1); s5_sar(T, T, 1);
        if (!s5_fits(T, ew[2])) return LF_FLAG;
        // sign2 + W1 choice (contents == values here)
        u64 dq = ((T[0] ^ B[0]) >> 1) & 1;
        u64 zx = (T[0] ^ B[0]) >> 2;
        u64 z2 = (zx ^ dq) & 1, zc3 = ((zx >> 1) ^ dq) & 1, z4 = ((zx >> 2) ^ dq) & 1;
        u64 sgT = T[4] >> 63, sgB = B[4] >> 63;
        u64 eq = sgT ^ sgB ^ dq;
        u64 wv = 0;
        if (z2 & eq) {
            const int K = LF_KT[t], L = K + c_lfUP; int h = (int)c_lfanch[t] - K; if (h < 0) h = 0;
            u64 a = s5_window(T, 0ULL - sgT, h, L), b = s5_window(B, 0ULL - sgB, h, L);
            wv = a > 4ULL * b;
        }
        u64 zero3 = ((z2 | zc3 | z4) == 0);
        u64 s3 = dq ^ 1ULL ^ zero3 ^ (z2 & eq & (wv ^ 1ULL));
        u64 dsel = s3 ^ dq;
        u64 k2 = dsel & (z2 ^ 1ULL);
        u64 k1 = dsel ^ (k2 & zc3);
        int e = (int)(k1 + 2 * k2);
        s5_addsub(T, T, B, (u32)s3); s5_sar(T, T, 1);
        if (!s5_fits(T, ew[3])) return LF_FLAG;
        if (e) s5_sar(T, T, e);
        if (!s5_fits(T, ew[4])) return LF_FLAG;
        s5_copy(r[ti], T);
    }
    bool p0 = s5_is_pm1(r[0]), p1 = s5_is_pm1(r[1]);
    if (leg == 0) return p0 ? (p1 ? LF_OK : LF_PHASE) : LF_FLAG;
    return (p0 && p1) ? LF_OK : LF_FLAG;
}

// ---------------------------------------------------------------- tier 1, limb-count specialised (v2)
// Tick t runs on NL(t) = ceil((maxw(t) + 2) / 64) limbs, maxw(t) = max(W(t), n0, n1, n2, nb, W(t+1)) (host table): every
// value the tick forms fits maxw + 1 bits while the tick's checks hold, and both rails leaving a clean tick fit W(t+1),
// so the representation is exact.  Upper limbs are kept as sign fill.  Decisions identical to lf_tier1.
__constant__ unsigned char c_lfnl[LF_MAXR];
template <int NL> __device__ __forceinline__ void n_addsub(u64 r[5], const u64 a[5], const u64 b[5], u32 sub) {
    u64 m = 0ULL - (u64)sub;
    if constexpr (NL == 1) {
        asm("{.reg .u32 c;\n\tadd.cc.u32 c, 0xffffffff, %3;\n\taddc.u64 %0, %1, %2;\n\t}"
            : "=l"(r[0]) : "l"(a[0]), "l"(b[0] ^ m), "r"(sub));
    } else if constexpr (NL == 2) {
        u64 r0, r1;
        asm("{.reg .u32 c;\n\tadd.cc.u32 c, 0xffffffff, %6;\n\taddc.cc.u64 %0, %2, %4;\n\taddc.u64 %1, %3, %5;\n\t}"
            : "=l"(r0), "=l"(r1) : "l"(a[0]), "l"(a[1]), "l"(b[0] ^ m), "l"(b[1] ^ m), "r"(sub));
        r[0] = r0; r[1] = r1;
    } else if constexpr (NL == 3) {
        u64 r0, r1, r2;
        asm("{.reg .u32 c;\n\tadd.cc.u32 c, 0xffffffff, %9;\n\taddc.cc.u64 %0, %3, %6;\n\taddc.cc.u64 %1, %4, %7;\n\taddc.u64 %2, %5, %8;\n\t}"
            : "=l"(r0), "=l"(r1), "=l"(r2) : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(b[0] ^ m), "l"(b[1] ^ m), "l"(b[2] ^ m), "r"(sub));
        r[0] = r0; r[1] = r1; r[2] = r2;
    } else if constexpr (NL == 4) {
        u64 r0, r1, r2, r3;
        asm("{.reg .u32 c;\n\tadd.cc.u32 c, 0xffffffff, %12;\n\taddc.cc.u64 %0, %4, %8;\n\taddc.cc.u64 %1, %5, %9;\n\taddc.cc.u64 %2, %6, %10;\n\taddc.u64 %3, %7, %11;\n\t}"
            : "=l"(r0), "=l"(r1), "=l"(r2), "=l"(r3) : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
              "l"(b[0] ^ m), "l"(b[1] ^ m), "l"(b[2] ^ m), "l"(b[3] ^ m), "r"(sub));
        r[0] = r0; r[1] = r1; r[2] = r2; r[3] = r3;
    } else {
        s5_addsub(r, a, b, sub);
    }
}
template <int NL> __device__ __forceinline__ void n_sar(u64 r[5], const u64 a[5], int k) {
#pragma unroll
    for (int i = 0; i < NL - 1; ++i) r[i] = (a[i] >> k) | (a[i + 1] << (64 - k));
    r[NL - 1] = (u64)((long long)a[NL - 1] >> k);
}
template <int NL> __device__ __forceinline__ bool n_fits(const u64 a[5], int e) {
    u64 fill = (u64)((long long)a[NL - 1] >> 63), acc = 0;
    const u64 *m = c_lfmask[e - 1];
#pragma unroll
    for (int j = 0; j < NL; ++j) acc |= (a[j] ^ fill) & ~m[j];
    return acc == 0;
}
template <int NL> __device__ __forceinline__ u64 n_window(const u64 v[5], u64 sgmask, int h, int n) {
    int wd = h >> 6, sh = h & 63;
    u64 lo = sgmask, hi = sgmask;   // limbs at/above NL read as sign fill (magnitude 0)
#pragma unroll
    for (int k = 0; k < NL; ++k) { lo = (k == wd) ? v[k] : lo; hi = (k == wd + 1) ? v[k] : hi; }
    lo ^= sgmask; hi ^= sgmask;
    u64 x = (lo >> sh) | (sh ? (hi << (64 - sh)) : 0ULL);
    return x & ((1ULL << n) - 1ULL);
}
template <int NL> __device__ __forceinline__ void n_signfill(u64 a[5]) {
    u64 fill = (u64)((long long)a[NL - 1] >> 63);
#pragma unroll
    for (int j = NL; j < 5; ++j) a[j] = fill;
}
// Limb-specialized hint: no dynamically indexed LFInt temporaries.
// The carry into p satisfies carry_p = a_p XOR b_p XOR sum_p.
// B is odd here, so the sign of its modular negation is complemented.
// Any exceptional even rail queues exact confirmation rather than rejection.
template<int NL> __device__ __forceinline__ int n_bit(const u64 a[5],int p){
    u64 v=0;int word=p>>6;
#pragma unroll
    for(int j=0;j<NL;j++)v=word==j?a[j]:v;
    return (int)((v>>(p&63))&1ULL);
}
template<int NL> __device__ __forceinline__ u64 n_wchoice(const u64 a[5],int w,int h){
    int bits=min(22,max(0,w-1-h));
    return bits?n_window<NL>(a,0ULL-(u64)n_bit<NL>(a,w-1),h,bits):0;
}
static constexpr int SF43_ST[][5]={{257,258,258,258,257},{257,258,258,258,257},{257,258,258,258,257},{257,258,258,257,256},{257,257,257,256,254},{256,256,255,254,253},{254,254,254,253,251},{253,253,252,251,250},{251,251,250,249,248},{249,249,248,247,246},{247,247,246,245,244},{245,245,244,243,242},{243,243,242,241,240},{241,241,240,239,238},{239,239,238,237,236},{237,237,236,235,234},{235,235,234,233,232},{233,233,232,231,230},{231,231,230,229,228},{229,229,228,227,226},{227,227,226,225,224},{225,225,224,223,222},{223,223,222,221,220},{220,220,220,219,218},{219,219,218,217,216},{217,217,216,215,214},{214,214,214,213,211},{213,213,212,211,210},{210,210,209,208,207},{208,208,207,206,205},{206,206,205,204,203},{204,204,203,202,201},{202,202,201,200,199},{200,200,199,198,197},{198,198,197,196,195},{195,195,194,194,193},{193,193,192,191,190},{191,191,190,189,188},{189,189,188,187,186},{187,187,186,185,184},{185,185,184,183,182},{183,183,182,181,180},{180,181,180,179,178},{178,178,177,176,175},{176,176,175,174,173},{174,174,173,172,171},{172,172,171,170,169},{169,169,169,168,167},{167,167,166,165,164},{165,165,164,163,162},{163,163,162,161,160},{161,161,160,159,158},{159,159,158,157,156},{157,157,156,155,154},{155,155,154,153,152},{153,153,152,151,150},{151,151,150,149,148},{148,148,147,147,146},{146,146,145,145,144},{144,144,143,143,141},{143,143,142,141,140},{140,140,139,138,137},{138,138,137,136,135},{136,136,135,134,133},{134,134,133,132,131},{131,131,130,130,129},{129,129,128,127,126},{127,127,126,125,124},{125,125,124,123,122},{122,122,121,120,119},{120,120,119,118,117},{117,117,117,116,115},{115,115,114,113,112},{113,113,112,111,110},{111,111,110,109,108},{109,109,108,107,106},{107,107,106,105,104},{104,104,103,102,101},{102,102,101,100,99},{100,100,99,98,97},{98,98,97,96,95},{96,96,95,94,92},{93,93,92,91,90},{91,91,90,89,88},{89,89,88,87,86},{87,87,86,85,84},{85,85,84,83,82},{82,82,81,81,80},{80,80,79,78,77},{78,78,77,76,75},{76,76,75,74,73},{73,73,72,72,71},{71,71,71,70,69},{69,69,68,67,66},{67,67,66,65,64},{65,65,64,63,62},{62,62,61,60,59},{60,60,59,58,57},{58,58,57,56,55},{56,56,55,54,53},{54,54,53,52,51},{51,51,50,49,48},{49,49,48,48,47},{47,47,46,45,44},{44,44,43,43,42},{42,42,41,40,39},{40,40,39,38,37},{38,38,37,36,35},{36,36,35,34,33},{34,34,33,32,31},{31,31,30,29,28},{29,29,28,28,27},{27,27,26,25,24},{24,24,23,23,22},{22,22,21,20,19},{20,20,19,18,17},{18,18,17,16,15},{16,16,15,14,13},{13,13,13,12,11},{11,11,10,9,8},{9,9,8,7,6},{6,3,3,6,5}};
static constexpr int SF43_AN[]={255,255,255,254,253,252,250,249,247,245,243,241,239,238,236,234,233,230,228,226,224,222,220,218,217,214,211,209,207,204,202,200,198,196,194,193,191,188,185,183,180,178,177,174,172,169,168,165,163,161,159,157,155,153,150,148,147,145,143,141,139,136,134,132,129,127,125,122,120,118,116,114,112,109,108,106,104,101,99,97,94,93,91,89,87,84,82,79,78,75,73,71,68,66,64,62,61,58,56,54,51,49,46,44,42,39,37,36,33,30,27,25,24,21,19,16,14,12,10,8,4,3};
static constexpr int SF43_KT[]={20,11,11,11,15,15,15,15,15,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,17,17,17,17,17,17,17,17,17,17,15,15,15,15,15,15,15,15,15,15,15,13,13,13,13,13,11,11,11,11,11,11,9,9,9,9,9,9,9,9,9};
static constexpr int SF43_NL[]={5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
template<int W> __host__ __device__ constexpr u64 sf43h_mask(){if constexpr(W<=0)return 0;else if constexpr(W>=64)return ~0ULL;else return (1ULL<<W)-1;}
template<int NL,int E> __device__ __forceinline__ bool sf43h_static_fits(const u64 a[5]){u64 fill=(u64)((long long)a[NL-1]>>63),acc=0;
 if constexpr(NL>0)acc|=(a[0]^fill)&~sf43h_mask<E-1-0>();
 if constexpr(NL>1)acc|=(a[1]^fill)&~sf43h_mask<E-1-64>();
 if constexpr(NL>2)acc|=(a[2]^fill)&~sf43h_mask<E-1-128>();
 if constexpr(NL>3)acc|=(a[3]^fill)&~sf43h_mask<E-1-192>();
 if constexpr(NL>4)acc|=(a[4]^fill)&~sf43h_mask<E-1-256>();
 return acc==0;}
template<int NL,int TT,int LEG> __device__ __forceinline__ bool sf43h_static_tick(u64 T[5],const u64 B[5],u64 seedv,int &record){
 constexpr int t=TT,leg=LEG;
 constexpr int w=SF43_ST[t][0],n0=SF43_ST[t][1],n1=SF43_ST[t][2],n2=SF43_ST[t][3],nb=SF43_ST[t][4];
 constexpr int wn=t+1<LF_R?SF43_ST[t+1][0]:w;constexpr bool pf=leg==0&&t>=LF_RDIV;
 constexpr int mn=w<wn?w:wn;constexpr int cb0=pf?(n0<mn?n0:mn):n0,cb1=pf?(n1<mn?n1:mn):n1,cb2=pf?(n2<mn?n2:mn):n2;
 if(t%6==0)record=0;
 if(!sf43h_static_fits<NL,w>(T)||!sf43h_static_fits<NL,w>(B))return false;
 u32 s0=0,s1=0;int ctop=0;
 if(t+1<LF_R){
  if(!sf43h_static_fits<NL,n0>(T)||!sf43h_static_fits<NL,cb0>(B))return false;
  s0=(u32)(((T[0]^B[0])>>1)&1);n_addsub<NL>(T,T,B,s0);n_sar<NL>(T,T,1);n_signfill<NL>(T);
  if(!sf43h_static_fits<NL,n1>(T)||!sf43h_static_fits<NL,cb1>(B))return false;
  if(!(B[0]&1))return false;
  s1=(u32)(((T[0]^B[0])>>1)&1);
  int asign=n_bit<NL>(T,n1-1),bsign=n_bit<NL>(B,n1-1)^(int)s1;
  n_addsub<NL>(T,T,B,s1);ctop=asign^bsign^n_bit<NL>(T,n1-1);
  n_sar<NL>(T,T,1);n_signfill<NL>(T);if(!sf43h_static_fits<NL,n2>(T))return false;
 }
 constexpr int bw=cb2;if(!sf43h_static_fits<NL,n2>(T)||!sf43h_static_fits<NL,bw>(B))return false;
 int dq=((T[0]^B[0])>>1)&1,z2=((T[0]^B[0])>>2&1)^dq;
 int zc3=((T[0]^B[0])>>3&1)^dq,z4=((T[0]^B[0])>>4&1)^dq;
 int h=max(SF43_AN[t]-(int)SF43_KT[t],0),bv=min(n2,bw),s3,e;
 int sgT=n_bit<NL>(T,n2-1),sgB=n_bit<NL>(B,bv-1);
 if(t==0){int tm=(1-(int)s1)&((seedv>>8)>(s0?0x5ULL:0x3ULL));int eq=(1-tm)^dq;
  s3=dq^1^int((z2|zc3|z4)==0)^(z2&eq);
  int ds=s3^dq,k2=ds&(1-z2);e=(ds^(k2&zc3))+2*k2;
 }else if(t+1==LF_R){int eq=sgT^sgB^dq;
  int L=SF43_KT[t]+2,na=min(L,max(0,n2-1-h)),nbits=min(L,max(0,bv-1-h));
  u64 va=na?n_window<NL>(T,0ULL-sgT,h,na):0,vb=nbits?n_window<NL>(B,0ULL-sgB,h,nbits):0;
  s3=dq^1^int((z2|zc3|z4)==0)^(z2&eq);
  int ds=s3^dq,k2=ds&(1-z2);e=(ds^(k2&zc3))+2*k2;
 }else{
  int tw=max(n2,10),bw2=min(max(bw,10),tw),L=SF43_KT[t]+2;
  int na=min(L,max(0,tw-1-h)),nbits=min(L,max(0,bw2-1-h));
  sgT=n_bit<NL>(T,tw-1);sgB=n_bit<NL>(B,bw2-1);
  u64 va=na?n_window<NL>(T,0ULL-sgT,h,na):0,vb=nbits?n_window<NL>(B,0ULL-sgB,h,nbits):0;
  int field=(record>>(2*(t&1)))&3,pos=(t%6)/2+1;
  int spent=pos==1?0:pos==2?(field&1):((field^(field>>1))&1);
  gpu24_decide(T[0],B[0],sgT,sgB,s1,ctop,fa50_y<LEG,t>(va,vb),spent,s3,e);
  if(e>=4)record^=pos<<(2*(t&1));
 }
 n_addsub<NL>(T,T,B,(u32)s3);n_sar<NL>(T,T,1);n_signfill<NL>(T);
 if(!sf43h_static_fits<NL,nb>(T)||(e&&(T[0]&((1ULL<<e)-1)))||e>nb-2)return false;
 if(e)n_sar<NL>(T,T,e);n_signfill<NL>(T);
 if(!sf43h_static_fits<NL,wn>(T)||!sf43h_static_fits<NL,wn>(B))return false;
 return true;
}
template<int LEG,int T,int END> __device__ __forceinline__ bool sf43h_seq(u64 a[5],u64 b[5],u64 seedv,int &record){
 if constexpr(T>=END)return true;else{if constexpr(T&1){if(!sf43h_static_tick<SF43_NL[T],T,LEG>(b,a,seedv,record))return false;}else{if(!sf43h_static_tick<SF43_NL[T],T,LEG>(a,b,seedv,record))return false;}return sf43h_seq<LEG,T+1,END>(a,b,seedv,record);}}
template<int LEG,int LO,int HI> __device__ __forceinline__ bool sf43h_group(u64 a[5],u64 b[5],u64 seedv,int &record){return sf43h_seq<LEG,LO,HI>(a,b,seedv,record);}
template<int LEG> __device__ __forceinline__ int sf43h_static_hint(const u64 d[4]){u64 a[5],b[5];lf_seed(a,b,d);int record=0;u64 seedv=s5_window(b,0ULL-(b[4]>>63),244,12);
 if(!sf43h_group<LEG,0,6>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,6,12>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,12,18>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,18,24>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,24,30>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,30,36>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,36,42>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,42,48>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,48,54>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,54,60>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,60,66>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,66,72>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,72,78>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,78,84>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,84,90>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,90,96>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,96,102>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,102,108>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,108,114>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,114,120>(a,b,seedv,record))return LF_FLAG;
 if(!sf43h_group<LEG,120,122>(a,b,seedv,record))return LF_FLAG;
 bool p0=s5_is_pm1(a),p1=s5_is_pm1(b);if constexpr(LEG==0)return p0?(p1?LF_OK:LF_PHASE):LF_FLAG;else return(p0&&p1)?LF_OK:LF_FLAG;}
__device__ int lf_tier1_nl(const u64 d[4],int leg){return leg==0?sf43h_static_hint<0>(d):sf43h_static_hint<1>(d);}
