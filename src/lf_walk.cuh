#ifndef LF_STRIP_CONTRACT
#define LF_STRIP_CONTRACT 1
#endif
#ifndef LF_YP8_VETO
#define LF_YP8_VETO 0
#endif
#if LF_YP8_VETO != 0 && LF_YP8_VETO != 1
#error LF_YP8_VETO_must_be_zero_or_one
#endif
// Source-bound finite-width Leapfrog forward/reverse rails and ideal mod-p payload.
// Port of snapshotted lfexact.py. Approximate payload cells and phase are unmodeled.
// Return0=modeled classical pass,1=modeled failure. Model-contract misses
// reject as disclosed yield loss; no classical leg is delegated to the CPU.
#include "lf_tables.cuh"
#include "fa50_policy.cuh"
struct LFInt{u64 v[5];};
__device__ __forceinline__ int lfbit(const LFInt&a,int b){return (a.v[b>>6]>>(b&63))&1ULL;}
__device__ __forceinline__ u64 lfmask(int n){return n<=0?0ULL:n>=64?~0ULL:(1ULL<<n)-1;}
__device__ __forceinline__ LFInt lfadd(const LFInt&a,const LFInt&b){
 LFInt r;u64 c=0;
 #pragma unroll
 for(int i=0;i<5;i++){u64 t=a.v[i]+b.v[i],c1=t<a.v[i];u64 z=t+c,c2=z<t;r.v[i]=z;c=c1|c2;}return r;
}
__device__ __forceinline__ LFInt lfneg(const LFInt&a){LFInt b;u64 c=1;
 #pragma unroll
 for(int i=0;i<5;i++){u64 z=~a.v[i]+c;c=c&&z==0;b.v[i]=z;}return b;
}
__device__ __forceinline__ LFInt lfsub(const LFInt&a,const LFInt&b){return lfadd(a,lfneg(b));}
__device__ __forceinline__ LFInt lfshr(const LFInt&a,int s){if(!s)return a;LFInt r;
 #pragma unroll
 for(int i=0;i<4;i++)r.v[i]=(a.v[i]>>s)|(a.v[i+1]<<(64-s));
 r.v[4]=(u64)(((long long)a.v[4])>>s);return r;
}
__device__ __forceinline__ u64 lfwin(const LFInt&a,int width,int h,int L){
 int n=min(L,max(0,width-1-h));if(!n)return 0;
 int i=h>>6,s=h&63;u64 z=a.v[i]>>s;if(s&&i<4)z|=a.v[i+1]<<(64-s);
 u64 m=lfmask(n);return (z&m)^(lfbit(a,width-1)?m:0ULL);
}
__device__ __forceinline__ bool lfunit(const LFInt&a){bool pos=a.v[0]==1,neg=a.v[0]==~0ULL;
 #pragma unroll
 for(int i=1;i<5;i++){pos&=a.v[i]==0;neg&=a.v[i]==~0ULL;}return pos||neg;
}
__device__ __forceinline__ int lfbwidth(int n,int w,int wn,bool fused){return fused?min(n,min(w,wn)):n;}
// Registers use canonical signed320 values of their actual finite-width contents.
// Resize truncates then sign extends; add halves modulo the actual add width.
// Forward and reverse integer walk/payload checks are modeled here. Phase coins
// and complete native channel validation remain CPU obligations. Finite
// correspondence is empirically qualified, not a completeness guarantee.
__device__ __forceinline__ LFInt lfresize(const LFInt&a,int w){
 LFInt r;int sg=lfbit(a,w-1);
 #pragma unroll
 for(int i=0;i<5;i++){u64 mask=lfmask(w-64*i);r.v[i]=(a.v[i]&mask)|(sg?~mask:0ULL);}return r;
}
// Carry into bit p of the unsigned finite-width sum a+b.  This is the
// carry wire that the YP8 circuit exposes while forced move 2 is paused.
__device__ __forceinline__ int lfcarry_into(const LFInt&a,const LFInt&b,int p){
 u64 c=0;int full=p>>6,rem=p&63;
 #pragma unroll
 for(int i=0;i<5;i++){
  if(i>=full)break;
  u64 z=a.v[i]+b.v[i],c1=z<a.v[i];u64 q=z+c,c2=q<z;c=c1|c2;
 }
 if(!rem)return (int)c;
 u64 m=lfmask(rem),x=a.v[full]&m,y=b.v[full]&m;
 u64 z=x+y,c1=z<x,q=z+c,c2=q<z;
 return (c1|c2)||((q>>rem)&1ULL);
}
__device__ __forceinline__ int lft0m(const LFInt&seed1,int s0,int s1){
 // Source r1 is a bit-0-less rail in the circuit.  Its indices 243..254
 // are full numeric bits 244..255 and bit 256 is the sign.
 int sg=lfbit(seed1,256);u64 v=0;
 #pragma unroll
 for(int i=0;i<4;i++)v|=(u64)(lfbit(seed1,252+i)^sg)<<i;
 u64 c=s0?0x5ULL:0x3ULL;
 return (1-s1)&(v>c);
}
__device__ __forceinline__ void lfletters_plain(const LFInt&T,const LFInt&B,int tw,int bw,int tick,int t0m,
                                                 int&s3,int&k1,int&k2){
 int dq=lfbit(T,1)^lfbit(B,1),z2=lfbit(T,2)^lfbit(B,2)^dq;
 int zc3=lfbit(T,3)^lfbit(B,3)^dq,z4=lfbit(T,4)^lfbit(B,4)^dq;
 int eq=tick==0?((1-t0m)^dq):(lfbit(T,tw-1)^lfbit(B,min(tw-1,bw-1))^dq);
 int veto=1;
 if(tick!=0 && tick!=LF_R-1){int h=max(LF_ANCHORS[tick]-(int)LF_KT[tick],0);
  veto=!(lfwin(T,tw,h,(int)LF_KT[tick]+2)>4*lfwin(B,min(tw,bw),h,(int)LF_KT[tick]+2));}
 s3=dq^1^int(z2==0&&zc3==0&&z4==0)^(z2&eq&veto);
 int ds=s3^dq;k2=ds&(1-z2);k1=ds^(k2&zc3);
}
__device__ __forceinline__ void lfletters_yp8(const LFInt&T,const LFInt&B,int tw,int bw,int s1,int ctop,int tick,
                                               int&s3,int&k1,int&k2){
 int dq=lfbit(T,1)^lfbit(B,1),z2=lfbit(T,2)^lfbit(B,2)^dq;
 int zc3=lfbit(T,3)^lfbit(B,3)^dq,z4=lfbit(T,4)^lfbit(B,4)^dq;
 int eq=lfbit(T,tw-1)^lfbit(B,min(tw-1,bw-1))^dq;
 int h=max(LF_ANCHORS[tick]-(int)LF_KT[tick],0),bv=min(tw,bw);
 int y=lfwin(T,tw,h,(int)LF_KT[tick]+2)>2*lfwin(B,bv,h,(int)LF_KT[tick]+2);
 int notflipped=1^ctop^lfbit(B,bv-1)^s1;
 int added=1^ctop^lfbit(T,tw-1);
 int special=(z2&notflipped)^(((1-z2)&zc3)&added);
#if LF_YP8_VETO
 // Release yp8_logic in both directions: E=eq, S=notflipped, C=added.
 // The bound source has a wire-valued window compare at every YP8 pause.
 // Its reverse phase uses the same q' and the forward tape is replayed here.
 int qprime=eq&(y^notflipped);
 s3=dq^1^int(z2==0&&zc3==0&&z4==0)^(qprime&(z2^((1-z2)&zc3&added)));
#else
 s3=dq^1^int(z2==0&&zc3==0&&z4==0)^(eq&(1-y)&special);
#endif
 int ds=s3^dq;k2=ds&(1-z2);k1=ds^(k2&zc3);
}
// gpu24: literal Boolean decision from published a2_rails.rs::yp8_logic_a.
// Inputs are numeric rails; the source sign-extends both decision views to >=10.
__device__ __forceinline__ void gpu24_decide(u64 tv,u64 bv,int sgT,int sgB,
 int s1,int ctop,int y,int spent,int &s3,int &e){
 int dq=((tv^bv)>>1)&1;
 int z2=(((tv^bv)>>2)&1)^dq,zc3=(((tv^bv)>>3)&1)^dq;
 int p4=1,p5,p6,p7,p8;
 #pragma unroll
 for(int i=2;i<=4;i++)p4&=1-((((tv^bv)>>i)&1)^dq);
 int g=p4&(1-spent);
 p5=g &(1-((((tv^bv)>>5)&1)^dq));
 p6=p5&(1-((((tv^bv)>>6)&1)^dq));
 p7=p6&(1-((((tv^bv)>>7)&1)^dq));
 p8=p7&(1-((((tv^bv)>>8)&1)^dq));
 int E=sgT^sgB^dq,S=1^ctop^sgB^s1,C=1^ctop^sgT;
 int qp=E&(y^S),v2=qp&z2,v3=qp&((1-z2)&zc3)&C;
 int I=p4^g^p8,ds=1^I^v2^v3;
 s3=dq^ds;
 int k3=g^p8,k2=(1-z2)^p4^p6^p8^v3;
 int a1=(1-z2)&(1-zc3);
 int k1=ds^(1-z2)^a1^v3^g^p5^p6^p7;
 e=k1+2*k2+4*k3;
}
__device__ __forceinline__ void gpu24_letters(const LFInt&T,const LFInt&B,
 int tw,int bw,int s1,int ctop,int tick,int spent,int leg,int&s3,int&e){
 int ta=max(tw,10),ba=min(max(bw,10),ta);
 int h=max(LF_ANCHORS[tick]-(int)LF_KT[tick],0);
 int y=fa50_y_runtime(lfwin(T,ta,h,(int)LF_KT[tick]+2),lfwin(B,ba,h,(int)LF_KT[tick]+2),leg,tick);
 gpu24_decide(T.v[0],B.v[0],lfbit(T,ta-1),lfbit(B,ba-1),s1,ctop,y,spent,s3,e);
}

__device__ __forceinline__ void lfhalf(u64 a[4]){
 u64 t[4],c=0;if(a[0]&1)c=u4_addc(t,a,PMODP);else u4_copy(t,a);
 u4_shr1(a,t);a[3]|=c<<63;
}
__device__ void lfpay(u64 pp[2][4],int t,int letter,bool rev){
 int ti=t&1,si=1-ti,s[3]={letter&1,(letter>>1)&1,(letter>>2)&1};
 int e=(letter>>3)&7,nf=t==LF_R-1?0:2;
 if(rev){for(int i=0;i<e;i++)fe_add(pp[ti],pp[ti],pp[ti]);
  fe_add(pp[ti],pp[ti],pp[ti]);if(s[2])fe_add(pp[ti],pp[ti],pp[si]);else fe_sub(pp[ti],pp[ti],pp[si]);
  for(int j=nf-1;j>=0;j--){fe_add(pp[ti],pp[ti],pp[ti]);if(s[j])fe_add(pp[ti],pp[ti],pp[si]);else fe_sub(pp[ti],pp[ti],pp[si]);}
 }else{for(int j=0;j<nf;j++){if(s[j])fe_sub(pp[ti],pp[ti],pp[si]);else fe_add(pp[ti],pp[ti],pp[si]);lfhalf(pp[ti]);}
  if(s[2])fe_sub(pp[ti],pp[ti],pp[si]);else fe_add(pp[ti],pp[ti],pp[si]);lfhalf(pp[ti]);
  for(int i=0;i<e;i++)lfhalf(pp[ti]);}
}
__device__ int lfwalk(const u64 den[4],int leg,const u64 expected[4],u64 *trace=nullptr){
 // sf43h full multiply model. Incomplete correspondence is disclosed as yield loss.
 /* sf43h: multiplication modeled on GPU, including reverse restoration. */
 int uncertain=0;
 LFInt d={{den[0],den[1],den[2],den[3],0}},p={{0xfffffffefffffc2fULL,~0ULL,~0ULL,~0ULL,0}};
 LFInt r[2];r[0]=(den[0]&1)?d:lfsub(d,p);
 r[1]=lfshr(lfbit(r[0],1)?lfadd(r[0],p):lfsub(r[0],p),1);
 LFInt seed0=r[0],seed1=r[1];unsigned char tape[LF_R]={0}; unsigned char fields[44]={0};u64 pp[2][4]={{1,0,0,0},{0,0,0,0}};u4_copy(pp[1],pp[0]);lfhalf(pp[1]);
 #pragma unroll 1
 for(int t=0;t<LF_R;t++){
  int w=LF_STEPS[t][0],n0=LF_STEPS[t][1],n1=LF_STEPS[t][2],n2=LF_STEPS[t][3],nb=LF_STEPS[t][4];
  int wn=t<LF_R-1?LF_STEPS[t+1][0]:w;bool pf=leg==0&&t>=LF_RDIV;
  int ti=t&1,si=1-ti;LFInt T=r[ti],B=r[si],decisionT,decisionB;int decisionbw=0;int letter=0,s0=0,s1=0,ctop=0;
  if(t<LF_R-1){
   T=lfresize(T,n0);int bw0=lfbwidth(n0,w,wn,pf);B=lfresize(B,bw0);
   s0=lfbit(T,1)^lfbit(B,1);letter|=s0;
   LFInt tot0=lfresize(s0?lfsub(T,B):lfadd(T,B),n0);if((tot0.v[0]&3)!=2){if(!trace)return 1;if(!uncertain)uncertain=t+1;}
   T=lfshr(tot0,1);if(pf){T=lfresize(T,min(n0-1,n1));B=lfresize(B,min(bw0,min(w,wn)));}

   T=lfresize(T,n1);int bw1=lfbwidth(n1,w,wn,pf);B=lfresize(B,bw1);
   s1=lfbit(T,1)^lfbit(B,1);letter|=s1<<1;
   LFInt badd=lfresize(B,n1);if(s1)badd=lfneg(badd);
   ctop=lfcarry_into(T,badd,n1-1);
   LFInt tot1=lfresize(s1?lfsub(T,B):lfadd(T,B),n1);if((tot1.v[0]&3)!=2){if(!trace)return 1;if(!uncertain)uncertain=t+1;}
   T=lfshr(tot1,1);decisionT=T;decisionB=B;decisionbw=bw1;if(pf){T=lfresize(T,min(n1-1,n2));B=lfresize(B,min(bw1,min(w,wn)));}
  }
  T=lfresize(T,n2);int bw=lfbwidth(n2,w,wn,pf);B=lfresize(B,bw);
  int s3,k1,k2;
  if(t==0)lfletters_plain(T,B,n2,bw,t,lft0m(seed1,s0,s1),s3,k1,k2);
  else if(t==LF_R-1)lfletters_plain(T,B,n2,bw,t,0,s3,k1,k2);
  else {
   int field=fields[2*(t/6)+(t&1)],pos=(t%6)/2+1;
   int spent=pos==1?0:pos==2?(field&1):((field^(field>>1))&1);
   int ea=0;gpu24_letters(decisionT,decisionB,n1-1,decisionbw,s1,ctop,t,spent,leg,s3,ea);
   k1=ea&1;k2=(ea>>1)&1;
   if(ea&4)fields[2*(t/6)+(t&1)]^=pos;
   letter|=(ea&4)<<3;
  }
  int e=k1+2*k2+((letter>>3)&4);
  letter|=(s3<<2)|(k1<<3)|(k2<<4);tape[t]=letter;
  if(leg==0)lfpay(pp,t,letter,false);
  LFInt tot=lfresize(s3?lfsub(T,B):lfadd(T,B),n2);if(tot.v[0]&1){if(!trace)return 1;if(!uncertain)uncertain=t+1;}
  T=lfshr(tot,1);T=lfresize(T,nb);
  if(t>0 && t+1<LF_R && e>min(7,nb-2)){if(!trace)return 1;if(!uncertain)uncertain=t+1;}
  if(T.v[0]&lfmask(e)){if(!trace)return 1;if(!uncertain)uncertain=t+1;}
#if LF_STRIP_CONTRACT
  if(!((T.v[0]>>e)&1ULL)){if(!trace)return 1;if(!uncertain)uncertain=t+1;}
#endif
  T=lfresize(lfshr(T,e),nb);r[ti]=lfresize(T,wn);r[si]=lfresize(B,wn);
  if(trace){for(int j=0;j<5;j++){trace[t*13+j]=r[0].v[j];trace[t*13+5+j]=r[1].v[j];}
   trace[t*13+10]=letter;trace[t*13+11]=t>0&&t+1<LF_R?fields[2*(t/6)+(t&1)]:0;
   trace[t*13+12]=uncertain;}

 }
 u64 zero[4]={0,0,0,0};int sign0=lfbit(r[0],319),sign1=lfbit(r[1],319);
 if(leg==0){if(sign0)fe_sub(pp[0],zero,pp[0]);if(!u4_eq(pp[0],expected))return 1;}
 else{pp[0][0]=1;pp[0][1]=pp[0][2]=pp[0][3]=0;u4_copy(pp[1],pp[0]);
  if(sign0)fe_sub(pp[0],zero,pp[0]);if(sign1)fe_sub(pp[1],zero,pp[1]);}
 #pragma unroll 1
 for(int t=LF_R-1;t>=0;t--){
  int w=LF_STEPS[t][0],n0=LF_STEPS[t][1],n1=LF_STEPS[t][2],n2=LF_STEPS[t][3],nb=LF_STEPS[t][4];
  int wn=t<LF_R-1?LF_STEPS[t+1][0]:w;bool pay=leg==1&&t>=LF_RMUL;
  int ti=t&1,si=1-ti,letter=tape[t];
  int s0=letter&1,s1=(letter>>1)&1,s3=(letter>>2)&1,e=(letter>>3)&7;
  if(leg==1)lfpay(pp,t,letter,true);
  LFInt T=lfresize(r[ti],pay?min(nb,wn):w),B=lfresize(r[si],pay?min(w,wn):w);
  T=lfresize(T,nb);B=lfresize(B,lfbwidth(n2,w,wn,pay));
  if(e){u64 wrap=0;int sg=lfbit(T,nb-1);for(int j=0;j<e;j++)wrap|=(u64)lfbit(T,nb-e+j)<<j;
   for(int j=0;j<e;j++)T=lfadd(T,T);T.v[0]^=wrap^(sg?lfmask(e):0);T=lfresize(T,nb);}
  T=lfresize(T,n2-1);T=lfresize(s3?lfadd(lfadd(T,T),B):lfsub(lfadd(T,T),B),n2);
  if(t<LF_R-1){
   for(int j=1;j>=0;j--){int nk=j?n1:n0,sub=j?s1:s0;
    T=lfresize(T,nk-1);B=lfresize(B,lfbwidth(nk,w,wn,pay));
    T=lfresize(sub?lfadd(lfadd(T,T),B):lfsub(lfadd(T,T),B),nk);}
  }
  r[ti]=lfresize(T,w);r[si]=lfresize(B,w);
 }
 if(uncertain)return 1;
 if(leg==1&&!u4_eq(pp[0],expected))return 1;
 // Native hoisted reverse correspondence remains open. This model executes
 // the restored-state check below; over-rejection is disclosed yield loss.
 /* sf43h: multiplication modeled on GPU, including reverse restoration. */
 LFInt restored=lfresize(r[0],257);
 #pragma unroll
 for(int i=0;i<5;i++)if(restored.v[i]!=seed0.v[i])return 1;
 return 0;
}
__device__ __forceinline__ bool lffit(const LFInt&a,int lo,int hi){
 int sg=lfbit(a,lo-1);
 #pragma unroll
 for(int i=0;i<5;i++){u64 m=lfmask(hi-64*i)&~lfmask(lo-1-64*i);if((a.v[i]&m)!=(sg?m:0ULL))return false;}return true;
}
__device__ __forceinline__ bool lftr(const LFInt&a,int from,int to){return to>from&&!lffit(a,from,to);}
// The hint NEVER makes a rejection decision. Only hint failures run the
// finite-register confirmation. This keeps recoverable-width cases admitted.
__device__ bool lfwalk_fast_hint(const u64 den[4],int leg){
 LFInt d={{den[0],den[1],den[2],den[3],0}},p={{0xfffffffefffffc2fULL,~0ULL,~0ULL,~0ULL,0}};
 LFInt r[2];r[0]=(den[0]&1)?d:lfsub(d,p);
 r[1]=lfshr(lfbit(r[0],1)?lfadd(r[0],p):lfsub(r[0],p),1);
 #pragma unroll 1
 for(int t=0;t<LF_R;t++){
  int w=LF_STEPS[t][0],n0=LF_STEPS[t][1],n1=LF_STEPS[t][2],n2=LF_STEPS[t][3],nb=LF_STEPS[t][4];
  int wn=t<LF_R-1?LF_STEPS[t+1][0]:w,wentry=t?w:max(w,260);
  bool pf=leg==0&&t>=LF_RDIV,pr=leg==1&&t>=LF_RMUL;int ti=t&1,si=1-ti;
  LFInt T=r[ti],B=r[si];
  LFInt T1=lfshr((lfbit(T,1)^lfbit(B,1))?lfsub(T,B):lfadd(T,B),1);
  LFInt T2=lfshr((lfbit(T1,1)^lfbit(B,1))?lfsub(T1,B):lfadd(T1,B),1);
  int bw0=lfbwidth(n0,w,wn,pf),bw1=lfbwidth(n1,w,wn,pf),bw2=lfbwidth(n2,w,wn,pf);
  int dq=lfbit(T2,1)^lfbit(B,1),z2=lfbit(T2,2)^lfbit(B,2)^dq,zc3=lfbit(T2,3)^lfbit(B,3)^dq,z4=lfbit(T2,4)^lfbit(B,4)^dq;
  int eq=lfbit(T2,n2-1)^lfbit(B,min(n2-1,bw2-1))^dq;
  int h=max(LF_ANCHORS[t]-(int)LF_KT[t],0),wv=lfwin(T2,n2,h,(int)LF_KT[t]+2)>4*lfwin(B,bw2,h,(int)LF_KT[t]+2);
  int s3=dq^1^int(z2==0&&zc3==0&&z4==0)^(z2&eq&(1-wv));
  int ds=s3^dq,k2=ds&(1-z2),k1=ds^(k2&zc3),e=k1+2*k2;
  LFInt T3=lfshr(s3?lfsub(T2,B):lfadd(T2,B),1),T4=lfshr(T3,e);
  bool bad=lftr(T,wentry,n0)||lftr(B,wentry,bw0)||lftr(B,bw0,n0);
  int a=n0-1;if(pf)a=min(a,n1);bad|=lftr(T1,a,n1)||lftr(B,bw0,bw1)||lftr(B,bw1,n1);
  a=n1-1;if(pf)a=min(a,n2);bad|=lftr(T2,a,n2)||lftr(B,bw1,bw2)||lftr(B,bw2,n2);
  bad|=lftr(T3,n2-1,nb);if(e)bad|=lftr(T4,nb-e,nb);
  bad|=lftr(T4,nb,wn)||lftr(B,bw2,wn);
  int bc;
  if(pr){int a4=min(nb,wn);bad|=lftr(T4,wn,a4)||lftr(T4,a4,nb);bc=min(w,wn);bad|=lftr(B,wn,bc);}
  else{bad|=lftr(T4,wn,w)||lftr(T4,w,nb)||lftr(B,wn,w);bc=w;}
  int rb2=lfbwidth(n2,w,wn,pr),rb1=lfbwidth(n1,w,wn,pr),rb0=lfbwidth(n0,w,wn,pr);
  bad|=lftr(T3,nb,n2-1)||lftr(B,bc,rb2)||lftr(B,rb2,n2);
  bad|=lftr(T2,n2,n1-1)||lftr(B,rb2,rb1)||lftr(B,rb1,n1);
  bad|=lftr(T1,n1,n0-1)||lftr(B,rb1,rb0)||lftr(B,rb0,n0);
  bad|=lftr(T,n0,w)||lftr(B,rb0,w);
  if(bad)return true;r[ti]=T4;
 }
 return !lfunit(r[0])||(leg==1&&!lfunit(r[1]));
}
// All 32 lanes participate, including exceptional inputs; zeros are replaced by1 for inversion.
__device__ void lfinverse(u64 out[4],const u64 den[4]){
 int lane=threadIdx.x&31;u64 x[4];u4_copy(x,den);if(u4_is_zero(x)){x[0]=1;x[1]=x[2]=x[3]=0;}
 u64 pre[4],suf[4];u4_copy(pre,x);u4_copy(suf,x);
 for(int d=1;d<32;d<<=1){u64 z[4];for(int j=0;j<4;j++)z[j]=__shfl_up_sync(0xffffffffu,pre[j],d);if(lane>=d)fe_mul(pre,pre,z);}
 for(int d=1;d<32;d<<=1){u64 z[4];for(int j=0;j<4;j++)z[j]=__shfl_down_sync(0xffffffffu,suf[j],d);if(lane+d<32)fe_mul(suf,suf,z);}
 u64 total[4],iv[4],left[4],right[4];for(int j=0;j<4;j++)total[j]=__shfl_sync(0xffffffffu,pre[j],31);
 if(lane==0)fe_inv(iv,total);for(int j=0;j<4;j++)iv[j]=__shfl_sync(0xffffffffu,iv[j],0);
 for(int j=0;j<4;j++){left[j]=__shfl_up_sync(0xffffffffu,pre[j],1);right[j]=__shfl_down_sync(0xffffffffu,suf[j],1);}
 if(lane==0){left[0]=1;left[1]=left[2]=left[3]=0;}if(lane==31){right[0]=1;right[1]=right[2]=right[3]=0;}
 fe_mul(out,iv,left);fe_mul(out,out,right);
}
__device__ void point_add_classical(u64 gx[4],u64 gy[4],const u64 tx[4],const u64 ty[4],const u64 ox[4],const u64 oy[4],u32*tape,u64*detail=nullptr){
 // Diagnostic ideal coordinate values; not a substitute for finite circuit payload.
 u64 d[4],di[4],num[4],lam[4],t[4];fe_sub(d,tx,ox);fe_inv(di,d);fe_sub(num,ty,oy);fe_mul(lam,num,di);
 fe_sqr(t,lam);fe_sub(t,t,tx);fe_sub(gx,t,ox);fe_sub(t,tx,gx);fe_mul(t,lam,t);fe_sub(gy,t,ty);
}
__device__ int shot_verdict(const u64 tx[4],const u64 ty[4],const u64 ox[4],const u64 oy[4],int ti,int oi,u32*tape,u64*obs=nullptr,u64*detail=nullptr){
 u64 d[4],di[4];fe_sub(d,tx,ox);lfinverse(di,d);
 if(ti||oi||u4_is_zero(d))return 0;
 if(lfwalk_fast_hint(d,0)){int div=lfwalk(d,0,di);if(div==2)return 0;if(div==1)return 1;}
 u64 num[4],lam[4],ex[4],tmp[4],d2[4];fe_sub(num,ty,oy);fe_mul(lam,num,di);
 fe_sqr(tmp,lam);fe_sub(tmp,tmp,tx);fe_sub(ex,tmp,ox);fe_sub(d2,ox,ex);
 if(u4_is_zero(d2))return 0;
 return lfwalk_fast_hint(d2,1)&&lfwalk(d2,1,d2)==1?1:0;
}
