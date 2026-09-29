// A/B microbenchmark: tiled vs scalar im2col, same stream, interleaved rounds.
// Shapes taken from the SAM3 neck census (3x3 convs on 256/512-ch f32 inputs).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define MAX_GRIDDIM_Y 65535
#define MAX_GRIDDIM_Z 65535
#define BLOCK 256
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while(0)
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

// ---- scalar reference kernel (verbatim logic from ggml im2col.cu) ----
__global__ void im2col_scalar(const float* x, __half* dst,
    long long IC, long long IW, long long IH, long long OH, long long OW, long long KW, long long KH,
    long long IC_IH_IW, long long IH_IW, long long N_OH, long long KH_KW, long long IC_KH_KW,
    int s0, int s1, int p0, int p1) {
    const long long i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= IC_KH_KW) return;
    const long long iic = i / KH_KW;
    const long long rem = i - iic * KH_KW;
    const long long ikh = rem / KW;
    const long long ikw = rem - ikh * KW;
    for (long long iow = blockIdx.y; iow < OW; iow += MAX_GRIDDIM_Y)
        for (long long iz = blockIdx.z; iz < N_OH; iz += MAX_GRIDDIM_Z) {
            const long long in = iz / OH, ioh = iz - in * OH;
            const long long iiw = iow * s0 + ikw - p0;   // d0 = d1 = 1
            const long long iih = ioh * s1 + ikh - p1;
            const long long off = ((in * OH + ioh) * OW + iow) * IC_KH_KW + i;
            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) dst[off] = __float2half(0.0f);
            else dst[off] = __float2half(x[iic * IC_IH_IW + in * IH_IW + iih * IW + iiw]);
        }
}

// ---- tiled kernel (copy of ggml im2col_tiled_kernel) ----
__global__ void im2col_tiled(const float* x, __half* dst,
    long long IC, long long IW, long long IH, long long OH, long long OW,
    long long KH, long long KW, long long KH_KW, long long IC_KH_KW,
    int s0, int s1, int p0, int p1,
    long long IC_B, long long TW, long long in_w,
    long long ic_blocks, long long tiles_x, long long rows) {
    extern __shared__ float smem[];
    const int c_total = (int)(IC_B * in_w);
    const int cw      = (int)in_w;
    const int out_span = (int)(IC_B * KH_KW);
    for (long long iz = blockIdx.z; iz < rows; iz += MAX_GRIDDIM_Z) {
        const int oh = (int)(iz % OH);
        const int n  = (int)(iz / OH);
        for (long long iy = blockIdx.y; iy < tiles_x; iy += MAX_GRIDDIM_Y) {
            const int ow0 = (int)(iy * TW);
            const int iw0 = ow0 * s0 - p0;
          for (long long ix = blockIdx.x; ix < ic_blocks; ix += MAX_GRIDDIM_Y) {
            const int ic0 = (int)(ix * IC_B);
            const int total_in = (int)KH * c_total;
            for (int t = threadIdx.x; t < total_in; t += blockDim.x) {
                const int kh = t / c_total;
                const int r  = t - kh * c_total;
                const int c  = r / cw;
                const int j  = r - c * cw;
                const int ic = ic0 + c;
                const int ih = oh * s1 + kh - p1;
                const int iw = iw0 + j;
                float v = 0.0f;
                if (ic < (int)IC && ih >= 0 && ih < (int)IH && iw >= 0 && iw < (int)IW)
                    v = x[((long long)n * IC + ic) * IH * IW + (long long)ih * IW + iw];
                smem[(kh * c_total) + r] = v;
            }
            __syncthreads();
            const int total_out = (int)TW * out_span;
            const long long row_base = ((long long)n * OH + oh) * OW * IC_KH_KW;
            for (int t = threadIdx.x; t < total_out; t += blockDim.x) {
                const int owl   = t / out_span;
                const int r     = t - owl * out_span;
                const int c     = r / (int)KH_KW;
                const int ikhkw = r - c * (int)KH_KW;
                const int ic    = ic0 + c;
                if (ic < (int)IC && ow0 + owl < (int)OW) {
                    const int kh  = ikhkw / (int)KW;
                    const int ikw = ikhkw - kh * (int)KW;
                    const float v = smem[kh * c_total + c * cw + owl * s0 + ikw];
                    dst[row_base + (long long)(ow0 + owl) * IC_KH_KW + ic * KH_KW + ikhkw] =
                        __float2half(v);
                }
            }
            __syncthreads();
          }
        }
    }
}

static void launch_scalar(const float* x, __half* dst, long long IC, long long IW, long long IH,
                          long long OH, long long OW, long long KW, long long KH, int s0, int s1, int p0, int p1) {
    const long long KH_KW = KW * KH, IC_KH_KW = IC * KH_KW, N = 1;
    const long long N_OH = N * OH;
    dim3 g((IC_KH_KW + BLOCK - 1) / BLOCK, MIN(OW, MAX_GRIDDIM_Y), MIN(N_OH, MAX_GRIDDIM_Z));
    im2col_scalar<<<g, BLOCK>>>(x, dst, IC, IW, IH, OH, OW, KW, KH, IH * IW, IC * IH * IW, N_OH, KH_KW, IC_KH_KW, s0, s1, p0, p1);
}
static void launch_tiled(const float* x, __half* dst, long long IC, long long IW, long long IH,
                         long long OH, long long OW, long long KW, long long KH, int s0, int s1, int p0, int p1) {
    const long long KH_KW = KW * KH, N = 1;
    long long TW = 32, IC_B = (BLOCK + KH_KW - 1) / KH_KW;
    if (IC_B > 32) IC_B = 32;
    long long in_w = (TW - 1) * s0 + KW;
    while (TW > 1 && IC_B * KH * in_w * 4 > 48 * 1024) { TW /= 2; in_w = (TW - 1) * s0 + KW; }
    size_t smem = (size_t)(IC_B * KH * in_w) * 4;
    long long icb = (IC + IC_B - 1) / IC_B, tx = (OW + TW - 1) / TW, rows = OH;
    dim3 g(MIN(icb, MAX_GRIDDIM_Y), MIN(tx, MAX_GRIDDIM_Y), MIN(rows, MAX_GRIDDIM_Z));
    im2col_tiled<<<g, BLOCK, smem>>>(x, dst, IC, IW, IH, OH, OW, KH, KW, KH_KW, IC * KH_KW,
                                     s0, s1, p0, p1, IC_B, TW, in_w, icb, tx, rows);
}

struct Shape { long long IC, IH, IW, OH, OW, KW, KH; int s0, s1, p0, p1; const char* tag; };

int main() {
    Shape shapes[] = {
        {256, 288, 288, 288, 288, 3, 3, 1, 1, 1, 1, "neck3x3 s1 [256,288,288]"},
        {512, 144, 144, 144, 144, 3, 3, 1, 1, 1, 1, "neck3x3 s1 [512,144,144]"},
        {256, 72, 72,   72, 72,   3, 3, 1, 1, 1, 1, "neck3x3 s1 [256,72,72]"},
        {1024,144, 144, 72, 72,  2, 2, 2, 2, 0, 0, "dconv2x2 s2 [1024,144,144]"},
    };
    for (auto& sh : shapes) {
        const long long in_elems = sh.IC * sh.IH * sh.IW;
        const long long out_elems = sh.OH * sh.OW * sh.IC * sh.KW * sh.KH;
        std::vector<float> x(in_elems);
        for (long long i = 0; i < in_elems; ++i) x[i] = (float)(i % 251) / 251.0f - 0.5f;
        float* dx; __half* dy; __half* dy2;
        CK(cudaMalloc(&dx, in_elems * 4)); CK(cudaMalloc(&dy, out_elems * 2)); CK(cudaMalloc(&dy2, out_elems * 2));
        cudaMemcpy(dx, x.data(), in_elems * 4, cudaMemcpyHostToDevice);
        // correctness
        launch_scalar(dx, dy, sh.IC, sh.IW, sh.IH, sh.OH, sh.OW, sh.KW, sh.KH, sh.s0, sh.s1, sh.p0, sh.p1);
        CK(cudaDeviceSynchronize());
        printf("scalar ok\n");
        launch_tiled(dx, dy2, sh.IC, sh.IW, sh.IH, sh.OH, sh.OW, sh.KW, sh.KH, sh.s0, sh.s1, sh.p0, sh.p1);
        CK(cudaDeviceSynchronize());
        printf("tiled ok\n");
        std::vector<__half> h1(out_elems), h2(out_elems);
        cudaMemcpy(h1.data(), dy, out_elems * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h2.data(), dy2, out_elems * 2, cudaMemcpyDeviceToHost);
        long long mism = 0;
        for (long long i = 0; i < out_elems; ++i) if (__half2float(h1[i]) != __half2float(h2[i])) ++mism;
        // timing: interleaved rounds
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ts, tt;
        for (int r = 0; r < 40; ++r) {
            cudaEventRecord(e0); launch_scalar(dx, dy, sh.IC, sh.IW, sh.IH, sh.OH, sh.OW, sh.KW, sh.KH, sh.s0, sh.s1, sh.p0, sh.p1); cudaEventRecord(e1);
            cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms);
            cudaEventRecord(e0); launch_tiled(dx, dy2, sh.IC, sh.IW, sh.IH, sh.OH, sh.OW, sh.KW, sh.KH, sh.s0, sh.s1, sh.p0, sh.p1); cudaEventRecord(e1);
            cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); tt.push_back(ms);
        }
        std::sort(ts.begin(), ts.end()); std::sort(tt.begin(), tt.end());
        float sm = ts[ts.size()/2], tm = tt[tt.size()/2];
        double gb = (in_elems * 4.0 + out_elems * 2.0) / 1e9;
        printf("%-28s mismatch=%lld  scalar %6.2f ms (%5.0f GB/s)  tiled %6.2f ms (%5.0f GB/s)  speedup %.2fx\n",
               sh.tag, mism, sm, gb / sm * 1e3, tm, gb / tm * 1e3, sm / tm);
        cudaFree(dx); cudaFree(dy); cudaFree(dy2);
    }
    return 0;
}
