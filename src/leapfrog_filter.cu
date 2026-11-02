// Source-bound Leapfrog prefilter. Phase and finite payload cells are unmodeled.
// Every survivor requires full trusted CPU C/P/A and executed-cost validation.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef unsigned char      u8;

__device__ u64 *d_native_inputs=nullptr;
__device__ unsigned char *d_observe_flags = nullptr;
__device__ u64 *d_observe_values = nullptr;
__device__ u64 *d_observe_detail = nullptr;
__device__ int d_observe_shot = -1;
__device__ unsigned long long d_evaluated_shots = 0;
#ifndef CLASSIC_DETAIL
#define CLASSIC_DETAIL 1
#endif

#define CUCHK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "leapfrog_gpu: CUDA error at %s:%d (%s): %s\n", __FILE__, __LINE__, #x, \
            cudaGetErrorString(_e)); exit(3);} } while (0)

// ---------------------------------------------------------------------------
// shape constants (mirror pingpong_filter.rs)
// ---------------------------------------------------------------------------
#define N_BITS       256
#define VALUE_WIDTH  259          // N + 3
#define NUM_TESTS    9024
#define RATE         136          // SHAKE256 rate
#define TAIL_OPS     96
#define ABSORB_BPO   49
#define CKP_SIZE     5064
#define MAX_ROUNDS   1024
#define MAX_BLK      64
#define FCV          0x1000003d1ULL
#define TAPE_WORDS   ((698+31)/32) // pinned walk tape has 22 words

#define LF_EXPECTED_OPS 10260493ULL
#include "checkpoint_binding.cuh"
// ===========================================================================
// Keccak-f[1600]                                     (pingpong_filter.rs:57)
// rho/pi is written out with literal lane indices so the 25-lane state stays
// in registers (a table-driven index would force it to local memory).
// ===========================================================================
static const u64 H_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL };
__constant__ u64 c_RC[24];

__device__ __forceinline__ u64 rotl64(u64 x, int n) { return (x << n) | (x >> (64 - n)); }

#define RHOPI(J, R) do { tmp = a[J]; a[J] = rotl64(t, R); t = tmp; } while (0)

__device__ __forceinline__ void d_keccakf(u64 a[25]) {
    for (int r = 0; r < 24; ++r) {
        u64 bc0 = a[0]^a[5]^a[10]^a[15]^a[20];
        u64 bc1 = a[1]^a[6]^a[11]^a[16]^a[21];
        u64 bc2 = a[2]^a[7]^a[12]^a[17]^a[22];
        u64 bc3 = a[3]^a[8]^a[13]^a[18]^a[23];
        u64 bc4 = a[4]^a[9]^a[14]^a[19]^a[24];
        u64 d0 = bc4 ^ rotl64(bc1, 1);
        u64 d1 = bc0 ^ rotl64(bc2, 1);
        u64 d2 = bc1 ^ rotl64(bc3, 1);
        u64 d3 = bc2 ^ rotl64(bc4, 1);
        u64 d4 = bc3 ^ rotl64(bc0, 1);
#pragma unroll
        for (int j = 0; j < 25; j += 5) {
            a[j]^=d0; a[j+1]^=d1; a[j+2]^=d2; a[j+3]^=d3; a[j+4]^=d4;
        }
        u64 t = a[1], tmp;
        RHOPI(10, 1); RHOPI( 7, 3); RHOPI(11, 6); RHOPI(17,10); RHOPI(18,15); RHOPI( 3,21);
        RHOPI( 5,28); RHOPI(16,36); RHOPI( 8,45); RHOPI(21,55); RHOPI(24, 2); RHOPI( 4,14);
        RHOPI(15,27); RHOPI(23,41); RHOPI(19,56); RHOPI(13, 8); RHOPI(12,25); RHOPI( 2,43);
        RHOPI(20,62); RHOPI(14,18); RHOPI(22,39); RHOPI( 9,61); RHOPI( 6,20); RHOPI( 1,44);
#pragma unroll
        for (int j = 0; j < 25; j += 5) {
            u64 b0=a[j],b1=a[j+1],b2=a[j+2],b3=a[j+3],b4=a[j+4];
            a[j  ] = b0 ^ ((~b1) & b2);
            a[j+1] = b1 ^ ((~b2) & b3);
            a[j+2] = b2 ^ ((~b3) & b4);
            a[j+3] = b3 ^ ((~b4) & b0);
            a[j+4] = b4 ^ ((~b0) & b1);
        }
        a[0] ^= c_RC[r];
    }
}
#undef RHOPI

// ===========================================================================
// secp256k1 field arithmetic                        (pingpong_filter.rs:139)
// ===========================================================================
__constant__ u64 c_PMOD[4];
static const u64 H_PMOD[4] = {0xfffffffefffffc2fULL, ~0ULL, ~0ULL, ~0ULL};
static const u64 H_GX[4] = {0x59f2815b16f81798ULL,0x029bfcdb2dce28d9ULL,0x55a06295ce870b07ULL,0x79be667ef9dcbbacULL};
static const u64 H_GY[4] = {0x9c47d08ffb10d4b8ULL,0xfd17b448a6855419ULL,0x5da4fbfc0e1108a8ULL,0x483ada7726a3c465ULL};

#ifdef __CUDA_ARCH__
#define PMODP c_PMOD
#else
#define PMODP H_PMOD
#endif

__host__ __device__ __forceinline__ u64 mulhi64(u64 a, u64 b) {
#ifdef __CUDA_ARCH__
    return __umul64hi(a, b);
#else
    // MSVC has no __int128; 32x32 split (host only, used for the comb table).
    u64 a0 = a & 0xffffffffULL, a1 = a >> 32, b0 = b & 0xffffffffULL, b1 = b >> 32;
    u64 p00 = a0*b0, p01 = a0*b1, p10 = a1*b0, p11 = a1*b1;
    u64 mid = (p00 >> 32) + (p01 & 0xffffffffULL) + (p10 & 0xffffffffULL);
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
#endif
}

__host__ __device__ __forceinline__ void u4_copy(u64 r[4], const u64 a[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = a[i];
}
__host__ __device__ __forceinline__ bool u4_is_zero(const u64 a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}
__host__ __device__ __forceinline__ bool u4_eq(const u64 a[4], const u64 b[4]) {
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
}
// pingpong_filter.rs:158
__host__ __device__ __forceinline__ bool u4_ge(const u64 a[4], const u64 b[4]) {
    if (a[3] != b[3]) return a[3] > b[3];
    if (a[2] != b[2]) return a[2] > b[2];
    if (a[1] != b[1]) return a[1] > b[1];
    if (a[0] != b[0]) return a[0] > b[0];
    return true;
}
// pingpong_filter.rs:169
__host__ __device__ __forceinline__ u64 u4_addc(u64 r[4], const u64 a[4], const u64 b[4]) {
#ifdef __CUDA_ARCH__
    u64 r0,r1,r2,r3,c;
    asm("add.cc.u64  %0, %5,  %9;\n\t"
        "addc.cc.u64 %1, %6, %10;\n\t"
        "addc.cc.u64 %2, %7, %11;\n\t"
        "addc.cc.u64 %3, %8, %12;\n\t"
        "addc.u64    %4,  0,   0;"
        : "=l"(r0),"=l"(r1),"=l"(r2),"=l"(r3),"=l"(c)
        : "l"(a[0]),"l"(a[1]),"l"(a[2]),"l"(a[3]),
          "l"(b[0]),"l"(b[1]),"l"(b[2]),"l"(b[3]));
    r[0]=r0; r[1]=r1; r[2]=r2; r[3]=r3;
    return c;
#else
    u64 c = 0;
    for (int i = 0; i < 4; ++i) {
        u64 s = a[i] + b[i];      u64 c1 = (s < a[i]);
        u64 s2 = s + c;           u64 c2 = (s2 < s);
        r[i] = s2;                c = c1 | c2;
    }
    return c;
#endif
}
// pingpong_filter.rs:180
__host__ __device__ __forceinline__ u64 u4_subb(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 brw = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 t  = a[i] - b[i];    u64 b1 = (a[i] < b[i]);
        u64 t2 = t - brw;        u64 b2 = (t < brw);
        r[i] = t2;               brw = b1 | b2;
    }
    return brw;
}
__host__ __device__ __forceinline__ void u4_not(u64 r[4], const u64 a[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = ~a[i];
}
__host__ __device__ __forceinline__ void u4_xor(u64 r[4], const u64 a[4], const u64 b[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = a[i] ^ b[i];
}
__host__ __device__ __forceinline__ void u4_shr1(u64 r[4], const u64 a[4]) {
    u64 t0 = (a[0]>>1)|(a[1]<<63), t1 = (a[1]>>1)|(a[2]<<63),
        t2 = (a[2]>>1)|(a[3]<<63), t3 = a[3]>>1;
    r[0]=t0; r[1]=t1; r[2]=t2; r[3]=t3;
}
__host__ __device__ __forceinline__ void u4_shl1(u64 r[4], const u64 a[4]) {
    u64 t3 = (a[3]<<1)|(a[2]>>63), t2 = (a[2]<<1)|(a[1]>>63),
        t1 = (a[1]<<1)|(a[0]>>63), t0 = a[0]<<1;
    r[0]=t0; r[1]=t1; r[2]=t2; r[3]=t3;
}
// pingpong_filter.rs:198
__host__ __device__ __forceinline__ void fe_norm(u64 r[4], const u64 a[4]) {
    if (u4_ge(a, PMODP)) u4_subb(r, a, PMODP); else u4_copy(r, a);
}
// pingpong_filter.rs:207
__host__ __device__ __forceinline__ void fe_add(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 t[4]; u64 c = u4_addc(t, a, b);
    if (c == 1 || u4_ge(t, PMODP)) u4_subb(r, t, PMODP); else u4_copy(r, t);
}
// pingpong_filter.rs:216
__host__ __device__ __forceinline__ void fe_sub(u64 r[4], const u64 a[4], const u64 b[4]) {
    u64 t[4]; u64 brw = u4_subb(t, a, b);
    if (brw == 1) u4_addc(r, t, PMODP); else u4_copy(r, t);
}
// pingpong_filter.rs:232
__host__ __device__ __forceinline__ void reduce512(u64 r[4], const u64 t[8]) {
    u64 m[5];
    u64 c = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 plo = t[4+i] * FCV;
        u64 phi = mulhi64(t[4+i], FCV);
        u64 s = plo + c;  u64 cc = (s < plo);
        m[i] = s;         c = phi + cc;
    }
    m[4] = c;
    u64 carry = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 s = m[i] + t[i];  u64 c1 = (s < m[i]);
        u64 s2 = s + carry;   u64 c2 = (s2 < s);
        m[i] = s2;            carry = c1 | c2;
    }
    m[4] = m[4] + carry;
    u64 rr[4] = { m[0], m[1], m[2], m[3] };
    u64 hi = m[4];
    while (hi != 0) {
        u64 addend[4] = { hi * FCV, mulhi64(hi, FCV), 0, 0 };
        hi = u4_addc(rr, rr, addend);
    }
    fe_norm(r, rr);
}
// pingpong_filter.rs:263
__device__ __forceinline__ void fe_mul_fast(u64 r[4], const u64 a[4], const u64 b[4]);
__host__ __device__ __forceinline__ void fe_mul(u64 r[4], const u64 a[4], const u64 b[4]) {
#if defined(__CUDA_ARCH__) && defined(FE_FAST_MUL)
    fe_mul_fast(r,a,b);return;
#endif
#ifdef __CUDA_ARCH__
    u32 aa[8],bb[8],tt[16];u64 packed[8];
#pragma unroll
    for(int i=0;i<8;i++){aa[i]=(u32)(a[i/2]>>((i&1)*32));bb[i]=(u32)(b[i/2]>>((i&1)*32));}
#pragma unroll
    for(int i=0;i<16;i++)tt[i]=0;
#pragma unroll
    for(int i=0;i<8;i++){
        u64 carry=0;
#pragma unroll
        for(int j=0;j<8;j++){
            u64 value=(u64)aa[i]*bb[j]+(u64)tt[i+j]+carry;
            tt[i+j]=(u32)value;carry=value>>32;
        }
        tt[i+8]=(u32)carry;
    }
#pragma unroll
    for(int i=0;i<8;i++)packed[i]=(u64)tt[2*i]|((u64)tt[2*i+1]<<32);
    reduce512(r,packed);
#else

    u64 t[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) t[i] = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        u64 carry = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            u64 lo = a[i] * b[j];
            u64 hi = mulhi64(a[i], b[j]);
            u64 s  = lo + t[i+j];  hi += (s < lo);
            u64 s2 = s + carry;    hi += (s2 < s);
            t[i+j] = s2;           carry = hi;
        }
        t[i+4] = carry;
    }
    reduce512(r, t);

#endif
}
#include "fe_fast.cuh"
// filter-speed-v3: device squaring by the dedicated 36-product PTX routine (value-identical, canonical).
__host__ __device__ __forceinline__ void fe_sqr(u64 r[4], const u64 a[4]) {
#ifdef __CUDA_ARCH__
#ifdef FE_FAST_SQR_OFF
    fe_mul(r, a, a);
#else
    fe_sqr_fast(r, a);
#endif
#else
    fe_mul(r, a, a);
#endif
}

// pingpong_filter.rs:280 -- a^(p-2); the exponent is the compile-time p-2.
__host__ __device__ void fe_inv(u64 r[4],const u64 a[4]) {
    u64 x2[4];u4_copy(x2,a);
    for(int i=0;i<1;i++)fe_sqr(x2,x2);
    fe_mul(x2,x2,a);
    u64 x3[4];u4_copy(x3,x2);
    for(int i=0;i<1;i++)fe_sqr(x3,x3);
    fe_mul(x3,x3,a);
    u64 x6[4];u4_copy(x6,x3);
    for(int i=0;i<3;i++)fe_sqr(x6,x6);
    fe_mul(x6,x6,x3);
    u64 x9[4];u4_copy(x9,x6);
    for(int i=0;i<3;i++)fe_sqr(x9,x9);
    fe_mul(x9,x9,x3);
    u64 x11[4];u4_copy(x11,x9);
    for(int i=0;i<2;i++)fe_sqr(x11,x11);
    fe_mul(x11,x11,x2);
    u64 x22[4];u4_copy(x22,x11);
    for(int i=0;i<11;i++)fe_sqr(x22,x22);
    fe_mul(x22,x22,x11);
    u64 x44[4];u4_copy(x44,x22);
    for(int i=0;i<22;i++)fe_sqr(x44,x44);
    fe_mul(x44,x44,x22);
    u64 x88[4];u4_copy(x88,x44);
    for(int i=0;i<44;i++)fe_sqr(x88,x88);
    fe_mul(x88,x88,x44);
    u64 x176[4];u4_copy(x176,x88);
    for(int i=0;i<88;i++)fe_sqr(x176,x176);
    fe_mul(x176,x176,x88);
    u64 x220[4];u4_copy(x220,x176);
    for(int i=0;i<44;i++)fe_sqr(x220,x220);
    fe_mul(x220,x220,x44);
    u64 x223[4];u4_copy(x223,x220);
    for(int i=0;i<3;i++)fe_sqr(x223,x223);
    fe_mul(x223,x223,x3);
    u64 t[4];u4_copy(t,x223);
    for(int i=0;i<23;i++)fe_sqr(t,t);
    fe_mul(t,t,x22);
    u64 u[4];u4_copy(u,t);
    for(int i=0;i<5;i++)fe_sqr(u,u);
    fe_mul(u,u,a);
    u64 v[4];u4_copy(v,u);
    for(int i=0;i<3;i++)fe_sqr(v,v);
    fe_mul(v,v,x2);
    u64 z[4];u4_copy(z,v);
    for(int i=0;i<2;i++)fe_sqr(z,z);
    fe_mul(z,z,a);
    u4_copy(r,z);
}

// ===========================================================================
// Jacobian EC                                       (pingpong_filter.rs:293)
// ===========================================================================
struct Jac { u64 x[4], y[4], z[4]; };

__host__ __device__ void jac_double(Jac *o, const Jac *p) {
    if (u4_is_zero(p->z)) {
        o->x[0]=1;o->x[1]=0;o->x[2]=0;o->x[3]=0;
        o->y[0]=1;o->y[1]=0;o->y[2]=0;o->y[3]=0;
        o->z[0]=0;o->z[1]=0;o->z[2]=0;o->z[3]=0; return;
    }
    u64 a[4],b[4],c[4],xb[4],d[4],e[4],f[4],x3[4],c8[4],y3[4],zc3[4],t[4],t2[4];
    fe_sqr(a, p->x);
    fe_sqr(b, p->y);
    fe_sqr(c, b);
    fe_add(xb, p->x, b);
    fe_sqr(t, xb); fe_add(t2, a, c); fe_sub(d, t, t2);
    fe_add(d, d, d);
    fe_add(t, a, a); fe_add(e, t, a);
    fe_sqr(f, e);
    fe_add(t, d, d); fe_sub(x3, f, t);
    { u64 c2[4], c4[4]; fe_add(c2, c, c); fe_add(c4, c2, c2); fe_add(c8, c4, c4); }
    fe_sub(t, d, x3); fe_mul(t2, e, t); fe_sub(y3, t2, c8);
    fe_mul(t, p->y, p->z); fe_add(zc3, t, t);
    u4_copy(o->x, x3); u4_copy(o->y, y3); u4_copy(o->z, zc3);
}

// madd-2007-bl with the degenerate cases            (pingpong_filter.rs:325)
__host__ __device__ void jac_add_affine(Jac *o, const Jac *p, const u64 qx[4], const u64 qy[4]) {
    if (u4_is_zero(p->z)) {
        u4_copy(o->x, qx); u4_copy(o->y, qy);
        o->z[0]=1;o->z[1]=0;o->z[2]=0;o->z[3]=0; return;
    }
    u64 z1z1[4],u2[4],s2[4],h[4],rr[4],t[4],t2[4];
    fe_sqr(z1z1, p->z);
    fe_mul(u2, qx, z1z1);
    fe_mul(t, qy, p->z); fe_mul(s2, t, z1z1);
    fe_sub(h, u2, p->x);
    fe_sub(rr, s2, p->y);
    if (u4_is_zero(h)) {
        if (u4_is_zero(rr)) { jac_double(o, p); return; }
        o->x[0]=1;o->x[1]=0;o->x[2]=0;o->x[3]=0;
        o->y[0]=1;o->y[1]=0;o->y[2]=0;o->y[3]=0;
        o->z[0]=0;o->z[1]=0;o->z[2]=0;o->z[3]=0; return;
    }
    u64 hh[4],ii[4],j[4],rd[4],v[4],x3[4],y1j[4],y3[4],zh[4],zc3[4];
    fe_sqr(hh, h);
    fe_add(t, hh, hh); fe_add(ii, t, t);
    fe_mul(j, h, ii);
    fe_add(rd, rr, rr);
    fe_mul(v, p->x, ii);
    fe_sqr(t, rd); fe_sub(t, t, j); fe_add(t2, v, v); fe_sub(x3, t, t2);
    fe_mul(y1j, p->y, j);
    fe_sub(t, v, x3); fe_mul(t, rd, t); fe_add(t2, y1j, y1j); fe_sub(y3, t, t2);
    fe_add(zh, p->z, h);
    fe_sqr(t, zh); fe_sub(t, t, z1z1); fe_sub(zc3, t, hh);
    u4_copy(o->x, x3); u4_copy(o->y, y3); u4_copy(o->z, zc3);
}

// ===========================================================================
#ifndef COMB_BITS
#define COMB_BITS 12
#endif
static_assert(COMB_BITS>=8 && COMB_BITS<=16,"measured comb window family");
constexpr int COMB_WINDOWS=(256+COMB_BITS-1)/COMB_BITS;
constexpr int COMB_POINTS=(1<<COMB_BITS)-1;
// device constants uploaded by the host
// ===========================================================================
__constant__ u64 c_state0[25];
__constant__ u64 c_stream[MAX_BLK * 17];
__constant__ u32 c_patch[TAIL_OPS];
__constant__ int c_nblk;
#include "lf_walk.cuh"
#include "sf43h_hint_flat.cuh"
#include "lf_hint_binding.cuh"
// Rare suspected failures are confirmed in a separate kernel. The point
// kernel does not carry finite-walk tape/payload state or reject from the hint.
struct LFRequest{u64 d1[4],di[4],d2[4];int slot,shot,hint;};
__device__ LFRequest* d_lf_requests;
__device__ int* d_lf_request_count;
__device__ int d_lf_capacity;
__device__ int d_lf_overflow=0;
__device__ void lf_enqueue(const u64 tx[4],const u64 ty[4],const u64 ox[4],const u64 oy[4],int ti,int oi,int slot,int shot){
 u64 d[4],di[4];fe_sub(d,tx,ox);lfinverse(di,d);if(ti||oi||u4_is_zero(d))return;
 int hint=lf_tier1_nl(d,0)==LF_FLAG?1:0;
 u64 num[4],lam[4],ex[4],tmp[4],d2[4];fe_sub(num,ty,oy);fe_mul(lam,num,di);
 fe_sqr(tmp,lam);fe_sub(tmp,tmp,tx);fe_sub(ex,tmp,ox);fe_sub(d2,ox,ex);
 if(!u4_is_zero(d2)&&lf_tier1_nl(d2,1)==LF_FLAG)hint|=2;
 if(!hint)return;int q=atomicAdd(d_lf_request_count,1);
 if(q>=d_lf_capacity){atomicExch(&d_lf_overflow,1);return;}
 LFRequest*r=&d_lf_requests[q];u4_copy(r->d1,d);u4_copy(r->di,di);u4_copy(r->d2,d2);r->slot=slot;r->shot=shot;r->hint=hint;
}
__global__ __launch_bounds__(128,2)
void lf_confirm_kernel(unsigned long long*counts,int*dead,int screen){
 int n=*d_lf_request_count;if(n>d_lf_capacity)n=d_lf_capacity;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  LFRequest r=d_lf_requests[i];int bad=0,uncertain=0;
  if(r.hint&1){int v=lfwalk(r.d1,0,r.di);bad=v==1;uncertain=v==2;}
  if(!bad&&!uncertain&&(r.hint&2))bad=lfwalk(r.d2,1,r.d2)==1;
  if(bad){atomicAdd(counts+r.slot,1ULL);if(screen)atomicExch(dead+r.slot,1);}
  if(d_observe_flags)d_observe_flags[(size_t)r.slot*NUM_TESTS+r.shot]=(unsigned char)bad;
 }
}
// ===========================================================================
// Fiat-Shamir producer kernels
// ===========================================================================
struct SqState { u64 st[25]; u32 lane; };

__device__ __forceinline__ void fs_init_one(u64 nonce, SqState *out);
__global__ void fs_init_kernel(const u64 *__restrict__ nonces, int n, SqState *sq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fs_init_one(nonces[i], &sq[i]);
}
__device__ __forceinline__ void fs_init_one(u64 nonce, SqState *out) {
    u64 st[25];
#pragma unroll
    for (int k = 0; k < 25; ++k) st[k] = c_state0[k];
    for (int b = 0; b < c_nblk; ++b) {
#pragma unroll
        for (int k = 0; k < 17; ++k) st[k] ^= c_stream[b * 17 + k];
        for (int p = 0; p < TAIL_OPS; ++p) {
            u32 e = c_patch[p];
            if ((int)(e & 0xffu) != b) continue;
            u32 ln = (e >> 8) & 0x1fu, sh = (e >> 13) & 0x3fu, nb = (e >> 19) & 0x3fu;
            u64 val = ((nonce >> nb) & 1ULL) << sh;
#pragma unroll
            for (int k = 0; k < 17; ++k) st[k] ^= (k == (int)ln) ? val : 0ULL;
        }
        d_keccakf(st);
    }
#pragma unroll
    for (int k = 0; k < 25; ++k) out->st[k] = st[k];
    out->lane = 0;
}

__global__ void fs_squeeze_kernel(SqState *sq, u64 *__restrict__ xof, int n,
                                  int nwords, int win_words, const int *__restrict__ dead,
                                  const int *__restrict__ active=nullptr) {
    int work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= n) return;
    int i = active ? active[work] : work;
    if (dead && dead[i]) return;
    u64 st[25];
#pragma unroll
    for (int k = 0; k < 25; ++k) st[k] = sq[i].st[k];
    u32 lane = sq[i].lane;
    u64 *out = xof + (size_t)i * (size_t)win_words;
    for (int w = 0; w < nwords; ++w) {
        if (lane == 17u) { d_keccakf(st); lane = 0; }
        u64 x = 0;
#pragma unroll
        for (int k = 0; k < 17; ++k) x = (k == (int)lane) ? st[k] : x;
        out[w] = x;
        ++lane;
    }
#pragma unroll
    for (int k = 0; k < 25; ++k) sq[i].st[k] = st[k];
    sq[i].lane = lane;
}

__global__ void init_active_kernel(int *__restrict__ active,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)active[i]=i;
}
__global__ void compact_active_kernel(const int *__restrict__ input,int n,
                                      const int *__restrict__ dead,int *__restrict__ output,
                                      int *__restrict__ output_count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int lane=threadIdx.x&31;
    int slot=i<n?input[i]:-1;
    unsigned keep=__ballot_sync(0xffffffffu,slot>=0 && !dead[slot]);
    int count=__popc(keep),base=0;
    if(lane==0 && count)base=atomicAdd(output_count,count);
    base=__shfl_sync(0xffffffffu,base,0);
    if(keep&(1u<<lane))output[base+__popc(keep&((1u<<lane)-1))]=slot;
}

// ===========================================================================
// model kernel -- one warp per nonce, lane = shot within a 32-shot pass
// ===========================================================================
__device__ __forceinline__ void shfl_up_fe(u64 r[4], const u64 a[4], int d) {
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = __shfl_up_sync(0xffffffffu, a[i], d, 32);
}
__device__ __forceinline__ void shfl_down_fe(u64 r[4], const u64 a[4], int d) {
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = __shfl_down_sync(0xffffffffu, a[i], d, 32);
}
__device__ __forceinline__ void shfl_idx_fe(u64 r[4], const u64 a[4], int src) {
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = __shfl_sync(0xffffffffu, a[i], src, 32);
}


__device__ __forceinline__ unsigned scalar_window(const u64 k[4],int wi) {
    int bit=wi*COMB_BITS,word=bit>>6,shift=bit&63;u64 low=0,high=0;
#pragma unroll
    for(int i=0;i<4;i++){low=i==word?k[i]:low;high=i==word+1?k[i]:high;}
    u64 digit=(low>>shift)|(shift?high<<(64-shift):0ULL);
    return (unsigned)(digit&COMB_POINTS);
}
__device__ __forceinline__ void comb_mul2(Jac *ja,Jac *jb,const u64 k1[4],const u64 k2[4],const u64* __restrict__ comb) {
    Jac a; a.x[0]=1;a.x[1]=0;a.x[2]=0;a.x[3]=0;
    a.y[0]=1;a.y[1]=0;a.y[2]=0;a.y[3]=0;
    a.z[0]=0;a.z[1]=0;a.z[2]=0;a.z[3]=0;Jac b=a;
    for(int wi=0;wi<COMB_WINDOWS;wi++) {
        unsigned d1=scalar_window(k1,wi),d2=scalar_window(k2,wi);
        if(d1){const u64*p=comb+(size_t)(wi*COMB_POINTS+d1-1)*8;Jac t;jac_add_affine(&t,&a,p,p+4);a=t;}
        if(d2){const u64*p=comb+(size_t)(wi*COMB_POINTS+d2-1)*8;Jac t;jac_add_affine(&t,&b,p,p+4);b=t;}
    }
    *ja=a;*jb=b;
}

// Occupancy knob: minBlocksPerMultiprocessor.  2 -> 202 regs / 8 warps per SM
// (the ptxas default here, zero spills).  Raising it trades registers for
// warps; measured on the 4090, 2 wins.  -maxrregcount is IGNORED when
// __launch_bounds__ is present, so this is the only lever that works.
#ifndef POINT_MINBLOCKS
#define POINT_MINBLOCKS 3
#endif
#ifndef MODEL_MINBLOCKS
#define MODEL_MINBLOCKS 2
#endif
#define SB_WORDS 13
__global__ __launch_bounds__(128, POINT_MINBLOCKS)
void point_kernel(const u64 *__restrict__ xof, const u64 *__restrict__ comb,
                  int n, int shot_lo, int shot_hi, int win_words,
                  unsigned long long *counts,int *dead,int screen,u64 *__restrict__ shotbuf,
                  const int *__restrict__ active = nullptr, const int *__restrict__ cursor = nullptr) {
    const int lane   = threadIdx.x & 31;
    const int wpb    = blockDim.x >> 5;
    const int nwarps = gridDim.x * wpb;
    const int warp0  = blockIdx.x * wpb + (threadIdx.x >> 5);
    for (int work = warp0; work < n; work += nwarps) {
        int slot = active ? active[work] : work;
        const u64 *base = xof + (size_t)slot * (size_t)win_words;
        const int sbase = cursor ? cursor[slot] : shot_lo, send = sbase + (shot_hi - shot_lo);
        for (int s0 = sbase; s0 < send; s0 += 64) {
            Jac a0, a1, b0, b1;
            {
                const u64 *w = base + (size_t)(s0 - sbase + lane) * 8;
                u64 k1[4] = { w[0], w[1], w[2], w[3] }, k2[4] = { w[4], w[5], w[6], w[7] };
                comb_mul2(&a0, &a1, k1, k2, comb);
            }
            {
                const u64 *w = base + (size_t)(s0 + 32 - sbase + lane) * 8;
                u64 k1[4] = { w[0], w[1], w[2], w[3] }, k2[4] = { w[4], w[5], w[6], w[7] };
                comb_mul2(&b0, &b1, k1, k2, comb);
            }
            const int ia0 = u4_is_zero(a0.z), ia1 = u4_is_zero(a1.z), ib0 = u4_is_zero(b0.z), ib1 = u4_is_zero(b1.z);
            if (ia0) { a0.z[0]=1;a0.z[1]=0;a0.z[2]=0;a0.z[3]=0; }
            if (ia1) { a1.z[0]=1;a1.z[1]=0;a1.z[2]=0;a1.z[3]=0; }
            if (ib0) { b0.z[0]=1;b0.z[1]=0;b0.z[2]=0;b0.z[3]=0; }
            if (ib1) { b1.z[0]=1;b1.z[1]=0;b1.z[2]=0;b1.z[3]=0; }
            // warp-wide Montgomery batch inversion of the 128 z values (one fe_inv)   [v7]
            u64 pa[4], pb[4], prod[4];
            fe_mul(pa, a0.z, a1.z); fe_mul(pb, b0.z, b1.z); fe_mul(prod, pa, pb);
            u64 inc[4];  u4_copy(inc, prod);
            for (int d = 1; d < 32; d <<= 1) { u64 o[4]; shfl_up_fe(o, inc, d); if (lane >= d) fe_mul(inc, inc, o); }
            u64 suf[4];  u4_copy(suf, prod);
            for (int d = 1; d < 32; d <<= 1) { u64 o[4]; shfl_down_fe(o, suf, d); if (lane + d < 32) fe_mul(suf, suf, o); }
            u64 total[4]; shfl_idx_fe(total, inc, 31);
            u64 tinv[4];  fe_inv(tinv, total);
            u64 preL[4];  shfl_up_fe(preL, inc, 1);
            if (lane == 0)  { preL[0]=1;preL[1]=0;preL[2]=0;preL[3]=0; }
            u64 sufR[4];  shfl_down_fe(sufR, suf, 1);
            if (lane == 31) { sufR[0]=1;sufR[1]=0;sufR[2]=0;sufR[3]=0; }
            u64 pinv[4]; fe_mul(pinv, tinv, preL); fe_mul(pinv, pinv, sufR);
            u64 inva[4], invb[4]; fe_mul(inva, pinv, pb); fe_mul(invb, pinv, pa);
            u64 txA[4], tyA[4], oxA[4], oyA[4], txB[4], tyB[4], oxB[4], oyB[4];
            {
                u64 zi[4], zz[4], zzz[4];
                fe_mul(zi, inva, a1.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
                fe_mul(txA, a0.x, zz); fe_mul(tyA, a0.y, zzz);
                fe_mul(zi, inva, a0.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
                fe_mul(oxA, a1.x, zz); fe_mul(oyA, a1.y, zzz);
                if (ia0) { txA[0]=0;txA[1]=0;txA[2]=0;txA[3]=0; tyA[0]=0;tyA[1]=0;tyA[2]=0;tyA[3]=0; }
                if (ia1) { oxA[0]=0;oxA[1]=0;oxA[2]=0;oxA[3]=0; oyA[0]=0;oyA[1]=0;oyA[2]=0;oyA[3]=0; }
                fe_mul(zi, invb, b1.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
                fe_mul(txB, b0.x, zz); fe_mul(tyB, b0.y, zzz);
                fe_mul(zi, invb, b0.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
                fe_mul(oxB, b1.x, zz); fe_mul(oyB, b1.y, zzz);
                if (ib0) { txB[0]=0;txB[1]=0;txB[2]=0;txB[3]=0; tyB[0]=0;tyB[1]=0;tyB[2]=0;tyB[3]=0; }
                if (ib1) { oxB[0]=0;oxB[1]=0;oxB[2]=0;oxB[3]=0; oyB[0]=0;oyB[1]=0;oyB[2]=0;oyB[3]=0; }
            }
            // harness skip rules (eval_circuit): t.x == o.x, or either point at infinity
            const int skipA = u4_eq(txA, oxA) || ia0 || ia1;
            const int skipB = u4_eq(txB, oxB) || ib0 || ib1;
            // second warp-wide batch inversion: d1 = tx - ox for both halves (one fe_inv per 64 shots)
            u64 dA[4], dB[4];
            fe_sub(dA, txA, oxA); fe_sub(dB, txB, oxB);
            if (skipA) { dA[0]=1;dA[1]=0;dA[2]=0;dA[3]=0; }
            if (skipB) { dB[0]=1;dB[1]=0;dB[2]=0;dB[3]=0; }
            u64 invdA[4], invdB[4];
            {
                u64 pr[4]; fe_mul(pr, dA, dB);
                u64 in2[4]; u4_copy(in2, pr);
                for (int d = 1; d < 32; d <<= 1) { u64 o[4]; shfl_up_fe(o, in2, d); if (lane >= d) fe_mul(in2, in2, o); }
                u64 su2[4]; u4_copy(su2, pr);
                for (int d = 1; d < 32; d <<= 1) { u64 o[4]; shfl_down_fe(o, su2, d); if (lane + d < 32) fe_mul(su2, su2, o); }
                u64 tot2[4]; shfl_idx_fe(tot2, in2, 31);
                u64 ti2[4]; fe_inv(ti2, tot2);
                u64 pl[4]; shfl_up_fe(pl, in2, 1); if (lane == 0) { pl[0]=1;pl[1]=0;pl[2]=0;pl[3]=0; }
                u64 sr[4]; shfl_down_fe(sr, su2, 1); if (lane == 31) { sr[0]=1;sr[1]=0;sr[2]=0;sr[3]=0; }
                u64 pi[4]; fe_mul(pi, ti2, pl); fe_mul(pi, pi, sr);
                fe_mul(invdA, pi, dB); fe_mul(invdB, pi, dA);
            }
            for (int half = 0; half < 2; half++) {
                const u64 *tx = half ? txB : txA, *ty = half ? tyB : tyA, *ox = half ? oxB : oxA, *oy = half ? oyB : oyA;
                const u64 *inv = half ? invdB : invdA;
                u64 d1[4], yin[4], lam[4], t[4], ex[4], d2[4];
                fe_sub(d1, tx, ox);
                fe_sub(yin, ty, oy);
                fe_mul(lam, yin, inv);
                fe_sqr(t, lam); fe_sub(t, t, tx); fe_sub(ex, t, ox);
                fe_sub(d2, ox, ex);
                u64 *o = shotbuf + ((size_t)work * (size_t)(shot_hi - shot_lo) + (size_t)(s0 - sbase + 32 * half + lane)) * SB_WORDS;
#pragma unroll
                for (int j = 0; j < 4; ++j) { o[j] = d1[j]; o[4 + j] = d2[j]; o[8 + j] = inv[j]; }
                o[12] = (u64)(half ? skipB : skipA);
                const int shot=s0+32*half+lane;
                if(d_native_inputs){const u64*vv[4]={tx,ty,ox,oy};for(int j=0;j<4;j++)for(int k=0;k<4;k++)d_native_inputs[((size_t)slot*NUM_TESTS+shot)*16+j*4+k]=vv[j][k];}
            }
        }
    }
}


// Independent hint kernel: one warp owns 32 consecutive shots from one slot.
// It only queues finite confirmations; it never rejects from the ideal hint.
__global__ __launch_bounds__(128,4)
void lf_hint_kernel(const u64*points,int n,int span,int lo,const int*active=nullptr,const int*cursor=nullptr){
 for(long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;idx<(long long)n*span;idx+=(long long)gridDim.x*blockDim.x){
  int work=idx/span,s=idx%span,slot=active?active[work]:work;
  int shot=(cursor?cursor[slot]:lo)+s;
  const u64*b=points+idx*SB_WORDS;int hint=0;
  if(!b[12]){hint=lf_tier1_nl(b,0)==LF_FLAG?1:0;if(!u4_is_zero(b+4)&&lf_tier1_nl(b+4,1)==LF_FLAG)hint|=2;}
  if(d_observe_flags)d_observe_flags[(size_t)slot*NUM_TESTS+shot]=0;
  if((threadIdx.x&31)==0)atomicAdd(&d_evaluated_shots,32ULL);
  if(!hint)continue;
  int q=atomicAdd(d_lf_request_count,1);if(q>=d_lf_capacity){atomicExch(&d_lf_overflow,1);continue;}
  LFRequest*r=&d_lf_requests[q];u4_copy(r->d1,b);u4_copy(r->d2,b+4);u4_copy(r->di,b+8);r->slot=slot;r->shot=shot;r->hint=hint;
 }
}

// ---- v5 refill pipeline kernels ----
__global__ void refill_kernel(const int *__restrict__ free_list, int n_new, u64 first_nonce, SqState *sq,
                              int *dead, int *cursor, unsigned long long *counts, u64 *nonce_of,
                              int *active, int active_base) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_new) return;
    int slot = free_list[k]; u64 nonce = first_nonce + (u64)k;
    fs_init_one(nonce, &sq[slot]);
    dead[slot] = 0; cursor[slot] = 0; counts[slot] = 0; nonce_of[slot] = nonce;
    active[active_base + k] = slot;
}
__global__ void advance_compact_kernel(const int *__restrict__ in, int n, const int *__restrict__ dead,
                                       int *cursor, const unsigned long long *__restrict__ counts,
                                       const u64 *__restrict__ nonce_of, int step,
                                       int *active_out, int *n_active, int *free_out, int *n_free,
                                       u64 *surv, int *n_surv, int surv_cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, lane = threadIdx.x & 31;
    int slot = i < n ? in[i] : -1;
    bool live = false, freed = false;
    if (slot >= 0) {
        if (dead[slot]) freed = true;
        else {
            int p = cursor[slot] + step; cursor[slot] = p;
            if (p >= NUM_TESTS) {
                freed = true;
                if (counts[slot] == 0) { int k = atomicAdd(n_surv, 1); if (k < surv_cap) surv[k] = nonce_of[slot]; }
            } else live = true;
        }
    }
    unsigned m = __ballot_sync(0xffffffffu, live); int c = __popc(m), b = 0;
    if (lane == 0 && c) b = atomicAdd(n_active, c);
    b = __shfl_sync(0xffffffffu, b, 0);
    if (live) active_out[b + __popc(m & ((1u << lane) - 1))] = slot;
    m = __ballot_sync(0xffffffffu, freed); c = __popc(m); b = 0;
    if (lane == 0 && c) b = atomicAdd(n_free, c);
    b = __shfl_sync(0xffffffffu, b, 0);
    if (freed) free_out[b + __popc(m & ((1u << lane) - 1))] = slot;
}

__global__ __launch_bounds__(128, MODEL_MINBLOCKS)
void model_kernel(const u64*points,int n,int shot_lo,int shot_hi,unsigned long long*counts,int*dead,int screen,
                  const int *__restrict__ active=nullptr) {
    int lane=threadIdx.x&31,wpb=blockDim.x>>5,nwarps=gridDim.x*wpb;
    int warp0=blockIdx.x*wpb+(threadIdx.x>>5);u32 tape[TAPE_WORDS];
    for(int work=warp0;work<n;work+=nwarps){
        int slot=active?active[work]:work;
        if(dead && dead[slot])continue;
        unsigned long long acc=0;int aborted=0;unsigned long long prior=counts[slot];
        for(int s0=shot_lo;s0<shot_hi;s0+=32){
            const u64*p=points+((size_t)slot*(shot_hi-shot_lo)+(s0-shot_lo+lane))*16;
            u64 tx[4],ty[4],ox[4],oy[4];
#pragma unroll
            for(int k=0;k<4;k++){tx[k]=p[k];ty[k]=p[k+4];ox[k]=p[k+8];oy[k]=p[k+12];}
            int ti=u4_is_zero(tx)&&u4_is_zero(ty),oi=u4_is_zero(ox)&&u4_is_zero(oy);
            int mm=shot_verdict(tx,ty,ox,oy,ti,oi,tape,d_observe_values?d_observe_values+((size_t)slot*NUM_TESTS+s0+lane)*8:nullptr,
                (d_observe_detail && slot==0 && s0+lane==d_observe_shot)?d_observe_detail:nullptr);
            if(d_observe_flags)d_observe_flags[(size_t)slot*NUM_TESTS+s0+lane]=(unsigned char)mm;
            if(lane==0)atomicAdd(&d_evaluated_shots,32ULL);
            unsigned bad=__ballot_sync(0xffffffffu,mm!=0);acc+=(unsigned)__popc(bad);
            if(screen && prior+acc>0){aborted=1;break;}
        }
        if(lane==0){counts[slot]=prior+acc;if(screen && aborted && dead)dead[slot]=1;}
    }
}


// ===========================================================================
// single-shot selftest kernel (1 thread) -- mirrors run_selftest()
// ===========================================================================
__global__ void selftest_kernel(const u64 *__restrict__ xof, const u64 *__restrict__ comb, u64 *out) {

    if (threadIdx.x || blockIdx.x) return;
    u32 tape[TAPE_WORDS];
    u64 k1[4] = { xof[0], xof[1], xof[2], xof[3] };
    u64 k2[4] = { xof[4], xof[5], xof[6], xof[7] };
    Jac a, b;
    comb_mul2(&a, &b, k1, k2, comb);
    u64 zi[4], zz[4], zzz[4], tx[4], ty[4], ox[4], oy[4];
    fe_inv(zi, a.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
    fe_mul(tx, a.x, zz); fe_mul(ty, a.y, zzz);
    fe_inv(zi, b.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
    fe_mul(ox, b.x, zz); fe_mul(oy, b.y, zzz);
    u64 d[4], di[4], lam[4], ex[4], ey[4], t[4];
    fe_sub(d, ox, tx); fe_inv(di, d); fe_sub(t, oy, ty); fe_mul(lam, t, di);
    fe_sqr(t, lam); fe_sub(t, t, tx); fe_sub(ex, t, ox);
    fe_sub(t, tx, ex); fe_mul(t, lam, t); fe_sub(ey, t, ty);
    u64 gx[4], gy[4];
    point_add_classical(gx, gy, tx, ty, ox, oy, tape);
    int o = 0;
#define EMIT(v) do { for (int i=0;i<4;++i) out[o+i] = (v)[i]; o += 4; } while (0)
    EMIT(k1); EMIT(k2); EMIT(tx); EMIT(ty); EMIT(ox); EMIT(oy);
    EMIT(ex); EMIT(ey); EMIT(gx); EMIT(gy);
#undef EMIT
}

// ===========================================================================
// ============================ HOST DRIVER ==================================
// ===========================================================================

// pingpong_filter.rs:366
static void h_batch_to_affine(const std::vector<Jac> &js, std::vector<u64> &out) {
    size_t n = js.size();
    std::vector<u64> prefix(n * 4);
    u64 acc[4] = {1,0,0,0};
    for (size_t i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) prefix[i*4+k] = acc[k];
        if (!u4_is_zero(js[i].z)) fe_mul(acc, acc, js[i].z);
    }
    u64 inv[4]; fe_inv(inv, acc);
    out.assign(n * 8, 0);
    for (size_t ii = n; ii-- > 0; ) {
        if (u4_is_zero(js[ii].z)) continue;                 // stays (0,0)
        u64 zinv[4]; fe_mul(zinv, inv, &prefix[ii*4]);
        fe_mul(inv, inv, js[ii].z);
        u64 z2[4], zc3[4];
        fe_sqr(z2, zinv); fe_mul(zc3, z2, zinv);
        fe_mul(&out[ii*8],     js[ii].x, z2);
        fe_mul(&out[ii*8 + 4], js[ii].y, zc3);
    }
}

// CombTable::build                                  (pingpong_filter.rs:393)
static void build_comb(std::vector<u64> &pts) {
    std::vector<Jac> jacs; jacs.reserve(COMB_WINDOWS * COMB_POINTS);
    u64 bx[4], by[4];
    u4_copy(bx, H_GX); u4_copy(by, H_GY);
    for (int w = 0; w < COMB_WINDOWS; ++w) {
        Jac acc; u4_copy(acc.x, bx); u4_copy(acc.y, by);
        acc.z[0]=1; acc.z[1]=0; acc.z[2]=0; acc.z[3]=0;
        jacs.push_back(acc);
        for (int d = 2; d <= COMB_POINTS; ++d) { Jac t; jac_add_affine(&t, &acc, bx, by); acc = t; jacs.push_back(acc); }
        Jac t2; u4_copy(t2.x, bx); u4_copy(t2.y, by); t2.z[0]=1;t2.z[1]=0;t2.z[2]=0;t2.z[3]=0;
        for (int i = 0; i < COMB_BITS; ++i) { Jac o; jac_double(&o, &t2); t2 = o; }
        std::vector<Jac> one(1, t2);
        std::vector<u64> a; h_batch_to_affine(one, a);
        u4_copy(bx, &a[0]); u4_copy(by, &a[4]);
    }
    h_batch_to_affine(jacs, pts);
}

// v7: GPU comb table. bases: COMB_WINDOWS affine points (x[4],y[4]); out: COMB_WINDOWS*COMB_POINTS*(x,y)
__global__ void comb_build_kernel(const u64 *__restrict__ bases, u64 *__restrict__ out) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= COMB_WINDOWS * COMB_POINTS) return;
    int w = e / COMB_POINTS, dsc = e % COMB_POINTS + 1;
    const u64 *bx = bases + 8 * w, *by = bx + 4;
    Jac acc; u4_copy(acc.x, bx); u4_copy(acc.y, by); acc.z[0]=1; acc.z[1]=0; acc.z[2]=0; acc.z[3]=0;
    int top = 31 - __clz(dsc);
    for (int i = top - 1; i >= 0; --i) {
        Jac t; jac_double(&t, &acc); acc = t;
        if ((dsc >> i) & 1) { jac_add_affine(&t, &acc, bx, by); acc = t; }
    }
    u64 zi[4], zz[4], zzz[4];
    fe_inv(zi, acc.z); fe_sqr(zz, zi); fe_mul(zzz, zz, zi);
    fe_mul(out + 8 * (size_t)e, acc.x, zz); fe_mul(out + 8 * (size_t)e + 4, acc.y, zzz);
}
static void comb_bases(std::vector<u64> &b) {
    b.assign(8 * COMB_WINDOWS, 0);
    u64 bx[4], by[4]; u4_copy(bx, H_GX); u4_copy(by, H_GY);
    for (int w = 0; w < COMB_WINDOWS; ++w) {
        u4_copy(&b[8 * w], bx); u4_copy(&b[8 * w + 4], by);
        Jac t2; u4_copy(t2.x, bx); u4_copy(t2.y, by); t2.z[0]=1;t2.z[1]=0;t2.z[2]=0;t2.z[3]=0;
        for (int i = 0; i < COMB_BITS; ++i) { Jac o; jac_double(&o, &t2); t2 = o; }
        std::vector<Jac> one(1, t2); std::vector<u64> a; h_batch_to_affine(one, a);
        u4_copy(bx, &a[0]); u4_copy(by, &a[4]);
    }
}

// Model::new                                        (pingpong_filter.rs:604)
static void usage() {
    fprintf(stderr,
      "usage: leapfrog_gpu --checkpoint <blob> --ops <count> --from <n> --to <n>\n"
      "       [--device N] [--batch N] [--window N] [--chunk N] [--screen] [--stream] [--async] [--survivor-file PATH]\n"
      "       [--selftest] [--verbose]\n"
      "       route is bound to the compiled Q1236 one-fold window-56 source; checkpoint and op count must match\n"
      "stdout: one \"<nonce> <modeled_rejection_count>\" line per nonce, ascending.\n"
      "Default is full provisional filter counts; not official correctness counts.\n"
      "--screen retains GPU C<=0 for trusted CPU9024; requires native qualification; exits after first mismatch at chunk boundary. --to is inclusive.\n"
      "THIS IS A SCREEN, NOT A VALIDATOR: phase-channel failures are invisible here, so\n"
      "every clean nonce still needs a CPU eval_circuit confirm.\n");
    exit(2);
}
static void hexbe(char *dst, const u64 a[4]) {
    sprintf(dst, "%016llx%016llx%016llx%016llx", a[3], a[2], a[1], a[0]);
}
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if(getenv("MODEL_VALUES_FILE") || getenv("MODEL_PROBE_SHOT")){
        fprintf(stderr,"Legacy coordinate probes are unsupported for Leapfrog. Use native inputs and per-shot flags.\n");return 2;
    }
#ifdef _WIN32
    // stdout must be BINARY: text mode would emit CRLF and break a byte-for-byte
    // `diff` against the CPU filter, which writes bare LF.
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    int reference_fixture = 0;
    const char *ckpath = 0;
    const char *survivor_path = 0;
    long long want_ops = -1, from = -1, to = -1;
    int device = 0, batch = 32768, window = 9024, chunk = 0, screen = 0, selftest = 0, verbose = 0, stream_inputs = 0, async_launches = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        #define NEXT() (((i + 1) < argc) ? argv[++i] : (usage(), (char*)0))
        if      (a == "--checkpoint") ckpath = NEXT();
        else if (a == "--ops")     want_ops = atoll(NEXT());
        else if (a == "--from")    from = atoll(NEXT());
        else if (a == "--to")      to = atoll(NEXT());
        else if (a == "--device")  device = atoi(NEXT());
        else if (a == "--batch")   batch = atoi(NEXT());
        else if (a == "--window")  window = atoi(NEXT());
        else if (a == "--chunk")   chunk = atoi(NEXT());
        else if (a == "--stream") stream_inputs = 1;
        else if (a == "--reference-fixture") {fprintf(stderr,"disabled legacy fixture\n");return 2;}
        else if (a == "--screen")  screen = 1;
        else if (a == "--async") async_launches = 1;
        else if (a == "--survivor-file") survivor_path = NEXT();
        else if (a == "--selftest") {fprintf(stderr,"Use check_controls.py for source-native qualification. Legacy model selftest disabled.\n");return 2;}
        else if (a == "--verbose" || a == "-v") verbose = 1;
        else if (a == "--allow-op-mismatch") {
            fprintf(stderr, "leapfrog_gpu: --allow-op-mismatch is disabled in the route-bound build.\n");
            return 2;
        }
        else usage();
        #undef NEXT
    }
    if (!ckpath) usage();
    if(screen && window!=9024){fprintf(stderr,"C0 hunting requires full9024\n");return 4;}
    if(reference_fixture && (screen || selftest || from!=9000851329353LL || to!=from || window!=9024)){fprintf(stderr,"Reference fixture requires exact single known nonce/full9024\n");return 4;}
    if (!selftest && (from < 0 || to < 0 || to < from)) usage();
    if (want_ops < 0) {
        fprintf(stderr, "leapfrog_gpu: --ops <count> is REQUIRED (it is the circuit fingerprint).\n");
        exit(2);
    }

    // ---- checkpoint ----
    FILE *f = fopen(ckpath, "rb");
    if (!f) { fprintf(stderr, "leapfrog_gpu: cannot open %s\n", ckpath); exit(2); }
    std::vector<u8> blob(CKP_SIZE + 1);
    size_t got = fread(blob.data(), 1, CKP_SIZE + 1, f);
    fclose(f);
    if (got != (size_t)CKP_SIZE) {
        fprintf(stderr, "leapfrog_gpu: %s is %zu bytes, expected %d\n", ckpath, got, CKP_SIZE); exit(2);
    }
    if (memcmp(blob.data(), "PPFSCKP1", 8) != 0) {
        fprintf(stderr, "leapfrog_gpu: bad magic in %s (expected PPFSCKP1)\n", ckpath); exit(2);
    }
    u64 n_ops = 0, residual_len = 0;
    memcpy(&n_ops, blob.data() + 8, 8);
    memcpy(&residual_len, blob.data() + 16, 8);
    u64 h_state[25];
    memcpy(h_state, blob.data() + 24, 200);
    const u8 *residual = blob.data() + 224;
    const u8 *tail     = blob.data() + 360;
    if (residual_len >= RATE) {
        fprintf(stderr, "leapfrog_gpu: residual_len %llu out of range\n", (unsigned long long)residual_len); exit(2);
    }
    u64 expected_ops = LF_EXPECTED_OPS;
    if (want_ops != (long long)expected_ops || n_ops != expected_ops) {
        fprintf(stderr,
            "leapfrog_gpu: REFUSING TO RUN -- route requires %llu ops; checkpoint=%llu --ops=%lld.\n",
            (unsigned long long)expected_ops,
            (unsigned long long)n_ops, want_ops);
        return 4;
    }
    if (verbose) fprintf(stderr, "checkpoint: %llu ops, residual_len=%llu\n",
                         (unsigned long long)n_ops, (unsigned long long)residual_len);

    if(memcmp(blob.data(),EXPECTED_CHECKPOINT,CKP_SIZE)!=0) {
        fprintf(stderr,"REFUSING TO RUN: checkpoint bytes differ from the compiled frozen packet\n");return 4;
    }
    // ---- padded absorb stream + per-nonce patch table ----
    size_t L = (size_t)residual_len + (size_t)TAIL_OPS * ABSORB_BPO;
    size_t full = L / RATE;
    size_t nblk = full + 1;
    if (nblk > MAX_BLK) { fprintf(stderr, "leapfrog_gpu: absorb stream too long (%zu blocks)\n", nblk); exit(2); }
    std::vector<u8> S(nblk * RATE, 0);
    memcpy(S.data(), residual, (size_t)residual_len);
    memcpy(S.data() + residual_len, tail, (size_t)TAIL_OPS * ABSORB_BPO);
    std::vector<u32> patch(TAIL_OPS);
    for (int k = 0; k < TAIL_OPS; ++k) {
        size_t off = (size_t)residual_len + (size_t)k * ABSORB_BPO + 17;
        for (int b = 0; b < 8; ++b) S[off + b] = 0;    // q_target is rewritten per nonce
        u32 blk = (u32)(off / RATE), ln = (u32)((off % RATE) / 8),
            sh = (u32)(8 * (off % 8)), nb = (u32)(k / 2);
        if (blk > 0xff || ln > 31 || sh > 63 || nb > 63) {
            fprintf(stderr, "leapfrog_gpu: patch encode overflow\n"); exit(2);
        }
        patch[k] = blk | (ln << 8) | (sh << 13) | (nb << 19);
    }
    S[L] ^= 0x1f;                                 // SHAKE256 pad
    S[full * RATE + RATE - 1] ^= 0x80;
    std::vector<u64> stream(nblk * 17);
    for (size_t b = 0; b < nblk; ++b)
        for (int k = 0; k < 17; ++k) memcpy(&stream[b*17 + k], &S[b*RATE + k*8], 8);

    // ---- device setup ----
    CUCHK(cudaSetDevice(device));
    cudaDeviceProp prop; CUCHK(cudaGetDeviceProperties(&prop, device));
    if (verbose) fprintf(stderr, "device %d: %s, %d SMs, sm_%d%d\n",
                         device, prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    CUCHK(cudaMemcpyToSymbol(c_RC,H_RC,sizeof(H_RC)));
    CUCHK(cudaMemcpyToSymbol(c_PMOD,H_PMOD,sizeof(H_PMOD)));
    if(verbose)fprintf(stderr,"route=gpu24-leverA122 conservative-finite-confirmation\n");
    CUCHK(cudaMemcpyToSymbol(c_state0, h_state, sizeof(h_state)));
    CUCHK(cudaMemcpyToSymbol(c_stream, stream.data(), stream.size() * 8));
    CUCHK(cudaMemcpyToSymbol(c_patch, patch.data(), patch.size() * 4));
    { int nb = (int)nblk; CUCHK(cudaMemcpyToSymbol(c_nblk, &nb, 4)); }

    double tc0 = now_s();
    const size_t comb_words = (size_t)COMB_WINDOWS * COMB_POINTS * 8;
    u64 *d_comb = 0;
    CUCHK(cudaMalloc(&d_comb, comb_words * 8));
    {
        std::vector<u64> bases; comb_bases(bases);
        u64 *d_bases = 0; CUCHK(cudaMalloc(&d_bases, bases.size() * 8));
        CUCHK(cudaMemcpy(d_bases, bases.data(), bases.size() * 8, cudaMemcpyHostToDevice));
        int nent = COMB_WINDOWS * COMB_POINTS;
        comb_build_kernel<<<(nent + 127) / 128, 128>>>(d_bases, d_comb);
        CUCHK(cudaGetLastError()); CUCHK(cudaDeviceSynchronize()); CUCHK(cudaFree(d_bases));
    }
    if (verbose) fprintf(stderr, "comb table: %zu points built on the GPU in %.3fs\n", comb_words / 8, now_s() - tc0);
    if (getenv("FILTER_COMB_CHECK")) {
        std::vector<u64> host; build_comb(host);
        std::vector<u64> dev(comb_words); CUCHK(cudaMemcpy(dev.data(), d_comb, comb_words * 8, cudaMemcpyDeviceToHost));
        bool same = host.size() == dev.size() && memcmp(host.data(), dev.data(), comb_words * 8) == 0;
        fprintf(stderr, "FILTER_COMB_CHECK entries=%zu identical=%d\n", comb_words / 8, (int)same);
        return same ? 0 : 1;
    }

    // ---- selftest ----
    if (selftest) {
        const u64 kats[4] = { 280927950051922ULL, 280927950051953ULL, 3004060ULL, 7ULL + (1ULL << 48) };
        u64 *d_nonces = 0, *d_xof = 0, *d_out = 0;
        SqState *d_sq = 0;
        CUCHK(cudaMalloc(&d_nonces, 4 * 8));
        CUCHK(cudaMalloc(&d_sq, 4 * sizeof(SqState)));
        CUCHK(cudaMalloc(&d_xof, 4 * 8 * 8));
        CUCHK(cudaMalloc(&d_out, 40 * 8));
        CUCHK(cudaMemcpy(d_nonces, kats, 4 * 8, cudaMemcpyHostToDevice));
        fs_init_kernel<<<1, 32>>>(d_nonces, 4, d_sq);
        CUCHK(cudaGetLastError());
        fs_squeeze_kernel<<<1, 32>>>(d_sq, d_xof, 4, 8, 8, (const int*)0);
        CUCHK(cudaDeviceSynchronize());
        CUCHK(cudaGetLastError());
        std::vector<u64> hx(32);
        CUCHK(cudaMemcpy(hx.data(), d_xof, 4 * 8 * 8, cudaMemcpyDeviceToHost));
        for (int i = 0; i < 4; ++i) {
            u8 bb[32]; memcpy(bb, &hx[i*8], 32);
            char s[80]; for (int j = 0; j < 32; ++j) sprintf(s + 2*j, "%02x", bb[j]);
            fprintf(stderr, "xof[0..32] nonce %-20llu = %s\n", (unsigned long long)kats[i], s);
        }
        selftest_kernel<<<1, 32>>>(d_xof, d_comb, d_out);
        CUCHK(cudaDeviceSynchronize());
        CUCHK(cudaGetLastError());
        std::vector<u64> ho(40);
        CUCHK(cudaMemcpy(ho.data(), d_out, 40 * 8, cudaMemcpyDeviceToHost));
        const char *nm[10] = { "k1","k2","tx","ty","ox","oy","ex","ey","gx","gy" };
        char h[80];
        for (int i = 0; i < 10; ++i) { hexbe(h, &ho[i*4]); fprintf(stderr, "%s = %s\n", nm[i], h); }
        int okx = memcmp(&ho[6*4], &ho[8*4], 32) == 0;
        int oky = memcmp(&ho[7*4], &ho[9*4], 32) == 0;
        fprintf(stderr, "gx %s\ngy %s\n", okx ? "OK" : "MISMATCH", oky ? "OK" : "MISMATCH");
        return (okx && oky) ? 0 : 1;
    }

    if ((unsigned long long)to >= (1ULL << 48)) {
        fprintf(stderr, "leapfrog_gpu: nonce range must stay below 2^48.\n");
        return 2;
    }
    if (survivor_path && !screen) {
        fprintf(stderr, "leapfrog_gpu: --survivor-file requires --screen.\n");
        return 2;
    }
    if (window <= 0 || (window & 63) || window > NUM_TESTS) {
        fprintf(stderr, "leapfrog_gpu: --window must be a positive multiple of 64, <= %d\n", NUM_TESTS); exit(2);
    }
    if (chunk == 0) chunk = 64;
    if(chunk>256){fprintf(stderr,"Queued filter supports chunks<=256.\n");return 2;}
    if (chunk < 32 || (chunk & 31) || chunk > window) {
        fprintf(stderr, "leapfrog_gpu: --chunk must be a positive multiple of 32, <= window\n"); exit(2);
    }
    if (batch <= 0) usage();
    chunk = (chunk + 63) & ~63; if (chunk > window) chunk = window;   // v4: the point kernel steps 64 shots

    long long total = to - from + 1;
    if ((long long)batch > total) batch = (int)total;

    const size_t full_words = (size_t)NUM_TESTS * 8;
    const size_t stride_words = stream_inputs ? (size_t)chunk*8 : full_words;
    u64 *d_nonces = 0, *d_xof = 0;
    unsigned long long *d_counts = 0;
    int *d_dead = 0;
    SqState *d_sq = 0;
    CUCHK(cudaMalloc(&d_nonces, (size_t)batch * 8));
    CUCHK(cudaMalloc(&d_sq, (size_t)batch * sizeof(SqState)));
    CUCHK(cudaMalloc(&d_xof, (size_t)batch * stride_words * 8));
    u64*d_points;CUCHK(cudaMalloc(&d_points,(size_t)batch*chunk*SB_WORDS*8));
    CUCHK(cudaMalloc(&d_counts, (size_t)batch * 8));
    CUCHK(cudaMalloc(&d_dead, (size_t)batch * 4));
    const bool compact_active=screen && stream_inputs && async_launches;
    int *d_active_a=nullptr,*d_active_b=nullptr,*d_active_count=nullptr;
    if(compact_active){
        CUCHK(cudaMalloc(&d_active_a,(size_t)batch*sizeof(int)));
        CUCHK(cudaMalloc(&d_active_b,(size_t)batch*sizeof(int)));
        CUCHK(cudaMalloc(&d_active_count,sizeof(int)));
    }
    if (verbose) fprintf(stderr, "xof buffer: %.1f MB (batch=%d model_window=%d chunk=%d) mode=%s ordering=%s async_launches=%d blocks_per_sm=%s\n",
                         (double)batch * stride_words * 8 / 1048576.0, batch, window, chunk,
                         screen ? "screen" : "exact", stream_inputs ? "stream-after-rejection" : "materialize-all-9024",async_launches,getenv("CLASSIC_BLOCKS_PER_SM")?getenv("CLASSIC_BLOCKS_PER_SM"):"72");

    const char* inputs_path=getenv("LF_INPUTS_FILE");FILE*inputs_file=nullptr;u64*d_inputs=nullptr;std::vector<u64>h_inputs;
    if(inputs_path){if(screen)return 2;CUCHK(cudaMalloc(&d_inputs,(size_t)batch*NUM_TESTS*128));CUCHK(cudaMemcpyToSymbol(d_native_inputs,&d_inputs,sizeof(d_inputs)));h_inputs.resize((size_t)batch*NUM_TESTS*16);inputs_file=fopen(inputs_path,"wb");if(!inputs_file)return 2;}
    const char* flags_path=getenv("MODEL_FLAGS_FILE");
    const char* values_path=getenv("MODEL_VALUES_FILE");
    const char* detail_shot=getenv("MODEL_PROBE_SHOT");
    unsigned char* d_flags=nullptr;FILE* flags_file=nullptr;
    std::vector<unsigned char> h_flags;
    u64* d_detail=nullptr;
    u64* d_values=nullptr;FILE* values_file=nullptr;std::vector<u64> h_values;
    if(values_path) {
        if(screen) return 2;
        CUCHK(cudaMalloc(&d_values,(size_t)batch*NUM_TESTS*64));
        CUCHK(cudaMemcpyToSymbol(d_observe_values,&d_values,sizeof(d_values)));
        h_values.resize((size_t)batch*NUM_TESTS*8);values_file=fopen(values_path,"wb");if(!values_file)return 2;
    }
    if(flags_path) {
        if(screen) {fprintf(stderr,"MODEL_FLAGS_FILE requires exact mode\n");return 2;}
        CUCHK(cudaMalloc(&d_flags,(size_t)batch*NUM_TESTS));
        CUCHK(cudaMemcpyToSymbol(d_observe_flags,&d_flags,sizeof(d_flags)));
        h_flags.resize((size_t)batch*NUM_TESTS);flags_file=fopen(flags_path,"wb");
        if(!flags_file) return 2;
    }
    if(detail_shot && !CLASSIC_DETAIL) {fprintf(stderr,"detailed probe disabled in this build\n");return 2;}
    if(detail_shot) {
        int shot=atoi(detail_shot);
        CUCHK(cudaMalloc(&d_detail,56*8));CUCHK(cudaMemset(d_detail,0,56*8));
        CUCHK(cudaMemcpyToSymbol(d_observe_detail,&d_detail,sizeof(d_detail)));
        CUCHK(cudaMemcpyToSymbol(d_observe_shot,&shot,sizeof(shot)));
    }
    const int TPB = 128;
    int blocks_per_sm=getenv("CLASSIC_BLOCKS_PER_SM")?atoi(getenv("CLASSIC_BLOCKS_PER_SM")):72;
    if(blocks_per_sm<1||blocks_per_sm>128)return 2;
    if(verbose)fprintf(stderr,"comb_bits=%d comb_windows=%d l2_cache_bytes=%d\n",COMB_BITS,COMB_WINDOWS,prop.l2CacheSize);
    int max_blocks = prop.multiProcessorCount * blocks_per_sm;         // 8 blocks/SM of 4 warps
    std::vector<u64> hn(batch);
    std::vector<unsigned long long> hc(batch);
    int lf_r=LF_R,lf_k=20,lf_up=2,lf_rdiv=LF_RDIV,lf_rmul=LF_RMUL;
    CUCHK(cudaMemcpyToSymbol(c_lfRDIV,&lf_rdiv,sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(c_lfRMUL,&lf_rmul,sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(c_lfR,&lf_r,sizeof(int)));CUCHK(cudaMemcpyToSymbol(c_lfK,&lf_k,sizeof(int)));CUCHK(cudaMemcpyToSymbol(c_lfUP,&lf_up,sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(c_lfeff,H_LFEFF,sizeof(H_LFEFF)));
    CUCHK(cudaMemcpyToSymbol(c_lfanch,H_LFANCH,sizeof(H_LFANCH)));
    CUCHK(cudaMemcpyToSymbol(c_lfnl,H_LFNL,sizeof(H_LFNL)));
    CUCHK(cudaMemcpyToSymbol(c_lfmask,H_LFMASK,sizeof(H_LFMASK)));
    LFRequest* lf_requests=nullptr;int* lf_request_count=nullptr;int lf_capacity=batch*chunk;
    CUCHK(cudaMalloc(&lf_requests,(size_t)lf_capacity*sizeof(LFRequest)));
    CUCHK(cudaMalloc(&lf_request_count,sizeof(int)));
    CUCHK(cudaMemcpyToSymbol(d_lf_requests,&lf_requests,sizeof(lf_requests)));
    CUCHK(cudaMemcpyToSymbol(d_lf_request_count,&lf_request_count,sizeof(lf_request_count)));
    CUCHK(cudaMemcpyToSymbol(d_lf_capacity,&lf_capacity,sizeof(lf_capacity)));
    FILE *survivors = 0;
    if (survivor_path) {
        survivors = fopen(survivor_path, "wbx");
        if (!survivors) {
            fprintf(stderr, "leapfrog_gpu: cannot append survivor file %s\n", survivor_path);
            return 2;
        }
    }
    cudaEvent_t phase_begin,phase_end;
    CUCHK(cudaEventCreate(&phase_begin));CUCHK(cudaEventCreate(&phase_end));
    double init_ms=0,gen_ms=0,model_ms=0;
    double t0 = now_s();

    const bool refill_mode = compact_active && survivors && !flags_path && !values_path && !detail_shot;
    if (refill_mode) {
        if (verbose) fprintf(stderr, "pipeline=refill slots=%d step=%d\n", batch, chunk);
        const int C = batch, SURV_CAP = 1 << 16;
        int *d_cursor = 0, *d_free = 0, *d_ctr = 0; u64 *d_nonce_of = 0, *d_surv = 0;
        CUCHK(cudaMalloc(&d_cursor, (size_t)C * 4)); CUCHK(cudaMalloc(&d_free, (size_t)C * 4));
        CUCHK(cudaMalloc(&d_ctr, 3 * 4)); CUCHK(cudaMalloc(&d_nonce_of, (size_t)C * 8));
        CUCHK(cudaMalloc(&d_surv, (size_t)SURV_CAP * 8));
        int *act_in = d_active_a, *act_out = d_active_b;
        long long remaining = total; u64 next = (u64)from; int reported = 0;
        int n0 = (int)(remaining < C ? remaining : C);
        init_active_kernel<<<(n0 + 255) / 256, 256>>>(d_free, n0); CUCHK(cudaGetLastError());
        refill_kernel<<<(n0 + 127) / 128, 128>>>(d_free, n0, next, d_sq, d_dead, d_cursor, d_counts, d_nonce_of, act_in, 0);
        CUCHK(cudaGetLastError());
        int nactive = n0; next += (u64)n0; remaining -= n0;
        CUCHK(cudaMemset(d_ctr, 0, 3 * 4));
        while (nactive > 0) {
            fs_squeeze_kernel<<<(nactive + 127) / 128, 128>>>(d_sq, d_xof, nactive, chunk * 8, (int)stride_words, d_dead, act_in);
            CUCHK(cudaGetLastError());
            int blocks = (nactive + TPB / 32 - 1) / (TPB / 32); if (blocks > max_blocks) blocks = max_blocks; if (blocks < 1) blocks = 1;
            CUCHK(cudaMemsetAsync(lf_request_count,0,sizeof(int)));
            point_kernel<<<blocks, TPB>>>(d_xof, d_comb, nactive, 0, chunk, (int)stride_words, d_counts, d_dead, 1, d_points, act_in, d_cursor);
            lf_hint_kernel<<<prop.multiProcessorCount*8,128>>>(d_points,nactive,chunk,0,act_in,d_cursor);
            CUCHK(cudaGetLastError());
            lf_confirm_kernel<<<prop.multiProcessorCount*2,128>>>(d_counts,d_dead,screen);
            CUCHK(cudaGetLastError());
            CUCHK(cudaGetLastError());
            CUCHK(cudaMemsetAsync(d_ctr, 0, 2 * 4));
            advance_compact_kernel<<<(nactive + 255) / 256, 256>>>(act_in, nactive, d_dead, d_cursor, d_counts, d_nonce_of, chunk,
                act_out, d_ctr, d_free, d_ctr + 1, d_surv, d_ctr + 2, SURV_CAP);
            CUCHK(cudaGetLastError());
            int h[3]; CUCHK(cudaMemcpy(h, d_ctr, 3 * 4, cudaMemcpyDeviceToHost));
            if (h[2] > SURV_CAP) { fprintf(stderr, "leapfrog_gpu: survivor buffer overflow\n"); return 5; }
            if (h[2] > reported) {
                std::vector<u64> found(h[2] - reported);
                CUCHK(cudaMemcpy(found.data(), d_surv + reported, found.size() * 8, cudaMemcpyDeviceToHost));
                if (fwrite(found.data(), 8, found.size(), survivors) != found.size()) { fprintf(stderr, "leapfrog_gpu: survivor spool write failed\n"); return 5; }
                fflush(survivors); reported = h[2];
            }
            int *t = act_in; act_in = act_out; act_out = t; nactive = h[0];
            int n_new = (int)(remaining < h[1] ? remaining : h[1]);
            if (n_new > 0) {
                refill_kernel<<<(n_new + 127) / 128, 128>>>(d_free, n_new, next, d_sq, d_dead, d_cursor, d_counts, d_nonce_of, act_in, nactive);
                CUCHK(cudaGetLastError());
                nactive += n_new; next += (u64)n_new; remaining -= n_new;
            }
        }
        CUCHK(cudaDeviceSynchronize());
    }
    for (long long b0 = from; !refill_mode && b0 <= to; b0 += batch) {
        int nb = (int)(((to - b0 + 1) < (long long)batch) ? (to - b0 + 1) : (long long)batch);
        for (int i = 0; i < nb; ++i) hn[i] = (u64)(b0 + i);
        CUCHK(cudaMemcpy(d_nonces, hn.data(), (size_t)nb * 8, cudaMemcpyHostToDevice));
        CUCHK(cudaMemset(d_counts, 0, (size_t)nb * 8));
        CUCHK(cudaMemset(d_dead, 0, (size_t)nb * 4));
        int nactive=nb,*active_in=d_active_a,*active_out=d_active_b;
        if(compact_active){
            init_active_kernel<<<(nb+255)/256,256>>>(active_in,nb);
            CUCHK(cudaGetLastError());
        }
        float elapsed=0;
        CUCHK(cudaEventRecord(phase_begin));
        fs_init_kernel<<<(nb+127)/128,128>>>(d_nonces,nb,d_sq);
        CUCHK(cudaGetLastError());CUCHK(cudaEventRecord(phase_end));CUCHK(cudaEventSynchronize(phase_end));
        CUCHK(cudaEventElapsedTime(&elapsed,phase_begin,phase_end));init_ms+=elapsed;
        if(!stream_inputs) {
            CUCHK(cudaEventRecord(phase_begin));
            fs_squeeze_kernel<<<(nb+127)/128,128>>>(d_sq,d_xof,nb,(int)full_words,(int)full_words,nullptr);
            CUCHK(cudaGetLastError());CUCHK(cudaEventRecord(phase_end));CUCHK(cudaEventSynchronize(phase_end));
            CUCHK(cudaEventElapsedTime(&elapsed,phase_begin,phase_end));gen_ms+=elapsed;
        }
        if(async_launches) {
            CUCHK(cudaEventRecord(phase_begin));
            for(int lo=0;lo<NUM_TESTS;lo+=chunk) {
                if(compact_active && nactive==0)break;
                int hi=lo+chunk;if(hi>NUM_TESTS)hi=NUM_TESTS;
                int nwork=compact_active?nactive:nb;
                const int*active=compact_active?active_in:nullptr;
                if(stream_inputs) {
                    fs_squeeze_kernel<<<(nwork+127)/128,128>>>(d_sq,d_xof,nwork,(hi-lo)*8,(int)stride_words,screen?d_dead:nullptr,active);
                    CUCHK(cudaGetLastError());
                }
                int blocks=(nwork+TPB/32-1)/(TPB/32);if(blocks>max_blocks)blocks=max_blocks;if(blocks<1)blocks=1;
                CUCHK(cudaMemsetAsync(lf_request_count,0,sizeof(int)));
                point_kernel<<<blocks,TPB>>>(d_xof+(stream_inputs?0:(size_t)lo*8),d_comb,nwork,lo,hi,(int)stride_words,d_counts,d_dead,screen,d_points,active);
                lf_hint_kernel<<<prop.multiProcessorCount*8,128>>>(d_points,nwork,hi-lo,lo,active);
                CUCHK(cudaGetLastError());
                lf_confirm_kernel<<<prop.multiProcessorCount*2,128>>>(d_counts,d_dead,screen);
                CUCHK(cudaGetLastError());
                CUCHK(cudaGetLastError());
#ifndef COMPACT_EVERY
#define COMPACT_EVERY 1
#endif
                if(compact_active && (((lo/chunk+1)%COMPACT_EVERY)==0 || hi==NUM_TESTS)){
                    CUCHK(cudaMemset(d_active_count,0,sizeof(int)));
                    compact_active_kernel<<<(nactive+255)/256,256>>>(active_in,nactive,d_dead,active_out,d_active_count);
                    CUCHK(cudaGetLastError());
                    CUCHK(cudaMemcpy(&nactive,d_active_count,sizeof(int),cudaMemcpyDeviceToHost));
                    int*temp=active_in;active_in=active_out;active_out=temp;
                }
            }
            CUCHK(cudaEventRecord(phase_end));CUCHK(cudaEventSynchronize(phase_end));
            CUCHK(cudaEventElapsedTime(&elapsed,phase_begin,phase_end));model_ms+=elapsed;
        } else for(int lo=0;lo<NUM_TESTS;lo+=chunk) {
            int hi=lo+chunk;if(hi>NUM_TESTS)hi=NUM_TESTS;
            if(stream_inputs) {
                CUCHK(cudaEventRecord(phase_begin));
                fs_squeeze_kernel<<<(nb+127)/128,128>>>(d_sq,d_xof,nb,(hi-lo)*8,(int)stride_words,screen?d_dead:nullptr);
                CUCHK(cudaGetLastError());CUCHK(cudaEventRecord(phase_end));CUCHK(cudaEventSynchronize(phase_end));
                CUCHK(cudaEventElapsedTime(&elapsed,phase_begin,phase_end));gen_ms+=elapsed;
            }
            int blocks=(nb+TPB/32-1)/(TPB/32);if(blocks>max_blocks)blocks=max_blocks;if(blocks<1)blocks=1;
            CUCHK(cudaEventRecord(phase_begin));
            CUCHK(cudaMemsetAsync(lf_request_count,0,sizeof(int)));
            point_kernel<<<blocks,TPB>>>(d_xof+(stream_inputs?0:(size_t)lo*8),d_comb,nb,lo,hi,(int)stride_words,d_counts,d_dead,screen,d_points);
            lf_hint_kernel<<<prop.multiProcessorCount*8,128>>>(d_points,nb,hi-lo,lo);
            CUCHK(cudaGetLastError());
            lf_confirm_kernel<<<prop.multiProcessorCount*2,128>>>(d_counts,d_dead,screen);
            CUCHK(cudaGetLastError());
                CUCHK(cudaGetLastError());
            CUCHK(cudaGetLastError());CUCHK(cudaEventRecord(phase_end));CUCHK(cudaEventSynchronize(phase_end));
            CUCHK(cudaEventElapsedTime(&elapsed,phase_begin,phase_end));model_ms+=elapsed;
        }
        CUCHK(cudaMemcpy(hc.data(), d_counts, (size_t)nb * 8, cudaMemcpyDeviceToHost));
        if(inputs_file){CUCHK(cudaMemcpy(h_inputs.data(),d_inputs,(size_t)nb*NUM_TESTS*128,cudaMemcpyDeviceToHost));if(fwrite(h_inputs.data(),NUM_TESTS*128,nb,inputs_file)!=(size_t)nb)return 5;}
        if(values_file) {
            CUCHK(cudaMemcpy(h_values.data(),d_values,(size_t)nb*NUM_TESTS*64,cudaMemcpyDeviceToHost));
            if(fwrite(h_values.data(),NUM_TESTS*64,nb,values_file)!=(size_t)nb)return 5;
        }
        if(flags_file) {
            CUCHK(cudaMemcpy(h_flags.data(),d_flags,(size_t)nb*NUM_TESTS,cudaMemcpyDeviceToHost));
            if(fwrite(h_flags.data(),NUM_TESTS,nb,flags_file)!=(size_t)nb) return 5;
        }
        if(d_detail) {
            u64 hd[56];CUCHK(cudaMemcpy(hd,d_detail,sizeof(hd),cudaMemcpyDeviceToHost));
            const char* names[]={"tx","ty","ox","oy","dx","sig","xu_div","lam","x_seam","x_sq","xu_mul","gy_fused","gx","gy"};
            for(int j=0;j<14;j++) {char h[65];hexbe(h,hd+4*j);fprintf(stderr,"PROBE %s %s\n",names[j],h);}
        }

        for (int i = 0; i < nb; ++i) {
            if (survivors) {
                if (hc[i] <= 0) {
                    u64 nonce = (u64)(b0 + i);
                    if (fwrite(&nonce, sizeof(nonce), 1, survivors) != 1) {
                        fprintf(stderr, "leapfrog_gpu: survivor spool write failed\n");
                        return 5;
                    }
                }
            } else {
                printf("%llu %llu\n", (unsigned long long)(b0 + i), hc[i]);
            }
        }
        if (survivors) fflush(survivors); else fflush(stdout);
    }
    if(inputs_file)fclose(inputs_file);
    if(flags_file) fclose(flags_file);
    if(values_file) fclose(values_file);
    int lf_overflow=0;CUCHK(cudaMemcpyFromSymbol(&lf_overflow,d_lf_overflow,sizeof(int)));
    if(lf_overflow){fprintf(stderr,"Finite confirmation queue overflow; discard output.\n");return 5;}
    unsigned long long evaluated=0;CUCHK(cudaMemcpyFromSymbol(&evaluated,d_evaluated_shots,8));
    fprintf(stderr,"evaluated_shots=%llu mean_shots=%.3f\n",evaluated,(double)evaluated/total);
    fprintf(stderr,"timing init_ms=%.3f gen_ms=%.3f model_ms=%.3f\n",init_ms,gen_ms,model_ms);
    double dt = now_s() - t0;
    if (verbose) fprintf(stderr, "screened %lld nonces in %.2fs = %.3f nonces/s\n",
                         total, dt, (double)total / (dt > 0 ? dt : 1e-9));
    if (survivors) fclose(survivors);
    return 0;
}
