"""Hand-written reference kernels — used by `anvil smoke` to exercise the whole
pipeline (write submission -> okbench validate -> compile -> bench) with NO LLM.

These are correct-but-not-fast: the point is to prove the plumbing works and to
give the agent a baseline to beat.
"""

# Correct shared-memory-tiled BF16 GEMM NT implementing the stable C ABI.
# C = alpha * A[M,K] @ B[N,K]^T + beta * C, fp32 accumulate.
SMOKE_GEMM_BF16_NT = r'''
#include "ops/gemm_bf16_nt/interface.h"

#define TILE 16

__global__ void anvil_smoke_gemm(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    long asm_, long ask, long bsn, long bsk, long csm, long csn,
    float alpha, float beta) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;   // i in [0, M)
  int col = blockIdx.x * TILE + threadIdx.x;   // j in [0, N)
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    int ak = k0 + threadIdx.x;
    As[threadIdx.y][threadIdx.x] = (row < M && ak < K)
        ? __bfloat162float(A[(long)row * asm_ + (long)ak * ask]) : 0.f;
    int colB = blockIdx.x * TILE + threadIdx.y;
    int bk = k0 + threadIdx.x;
    Bs[threadIdx.y][threadIdx.x] = (colB < N && bk < K)
        ? __bfloat162float(B[(long)colB * bsn + (long)bk * bsk]) : 0.f;
    __syncthreads();
    #pragma unroll
    for (int kk = 0; kk < TILE; ++kk)
      acc += As[threadIdx.y][kk] * Bs[threadIdx.x][kk];
    __syncthreads();
  }
  if (row < M && col < N) {
    long idx = (long)row * csm + (long)col * csn;
    float prev = (beta != 0.f) ? __bfloat162float(C[idx]) : 0.f;
    C[idx] = __float2bfloat16(alpha * acc + beta * prev);
  }
}

extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(
    const OpenKernelsGemmBF16NTArgs* args, cudaStream_t stream) {
  if (args == nullptr) return cudaErrorInvalidValue;
  dim3 block(TILE, TILE);
  dim3 grid((args->n + TILE - 1) / TILE, (args->m + TILE - 1) / TILE);
  anvil_smoke_gemm<<<grid, block, 0, stream>>>(
      args->a, args->b, args->c, args->m, args->n, args->k,
      args->a_stride_m, args->a_stride_k, args->b_stride_n, args->b_stride_k,
      args->c_stride_m, args->c_stride_n, args->alpha, args->beta);
  return cudaGetLastError();
}
'''

SMOKE_KERNELS = {"gemm_bf16_nt": SMOKE_GEMM_BF16_NT}
