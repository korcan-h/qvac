// TEMP DIAGNOSTIC (debug/cuda-capture-ci, do not merge).
//
// Standalone check of the Windows finetune crash, without ggml: during CUDA stream capture, does
// creating a graph memory allocation fail (cudaMallocAsync -> cudaErrorMemoryAllocation)?
//
//  1. plain cudaMallocAsync/cudaFreeAsync under capture, several sizes (no cuBLAS at all)
//  2. the cublasSgemm ggml issues for a LoRA-A matmul (T,N, m=8, n=128, k=2048|6144, TF32, 4 MiB
//     workspace), plus variants that might avoid cuBLAS allocating during capture
// Both suites run on a clean process, then again after recreating ggml's memory state (a 32 GiB
// address reservation with 1 GiB mapped, as its VMM pool does, plus ~3.5 GB of cudaMalloc).
//
// Prints one line per test; exits 1 if any capture failed.

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

static const char * cublas_status(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
        default:                             return "OTHER";
    }
}

struct node_counts {
    size_t total = 0;
    int    kernels = 0, allocs = 0, frees = 0, other = 0;
};

static node_counts count_nodes(cudaGraph_t graph) {
    node_counts c;
    if (cudaGraphGetNodes(graph, nullptr, &c.total) != cudaSuccess) {
        (void) cudaGetLastError();
        return c;
    }
    std::vector<cudaGraphNode_t> nodes(c.total);
    if (c.total == 0 || cudaGraphGetNodes(graph, nodes.data(), &c.total) != cudaSuccess) {
        (void) cudaGetLastError();
        return c;
    }
    for (cudaGraphNode_t n : nodes) {
        cudaGraphNodeType t;
        if (cudaGraphNodeGetType(n, &t) != cudaSuccess) {
            (void) cudaGetLastError();
            continue;
        }
        if (t == cudaGraphNodeTypeKernel) {
            c.kernels++;
        } else if (t == cudaGraphNodeTypeMemAlloc) {
            c.allocs++;
        } else if (t == cudaGraphNodeTypeMemFree) {
            c.frees++;
        } else {
            c.other++;
        }
    }
    return c;
}

// Runs body() under relaxed stream capture (as ggml does), then instantiates and launches the graph.
// body returns a short description of its own calls' results.
template <typename F> static void captured(const char * name, cudaStream_t stream, F body) {
    (void) cudaGetLastError();
    cudaError_t begin = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    if (begin != cudaSuccess) {
        printf("%-44s BeginCapture: %s\n", name, cudaGetErrorString(begin));
        g_failures++;
        (void) cudaGetLastError();
        return;
    }
    const std::string calls   = body();
    const cudaError_t pending = cudaGetLastError();  // what ggml's post-op check would see
    cudaGraph_t       graph   = nullptr;
    const cudaError_t end     = cudaStreamEndCapture(stream, &graph);
    node_counts       c;
    std::string       run = "not run";
    if (end == cudaSuccess && graph != nullptr) {
        c = count_nodes(graph);
        cudaGraphExec_t   exec = nullptr;
        const cudaError_t inst = cudaGraphInstantiate(&exec, graph, 0);
        if (inst == cudaSuccess) {
            const cudaError_t launch = cudaGraphLaunch(exec, stream);
            const cudaError_t sync   = cudaStreamSynchronize(stream);
            run = std::string("launch ") + cudaGetErrorName(launch) + ", sync " + cudaGetErrorName(sync);
            cudaGraphExecDestroy(exec);
        } else {
            run = std::string("instantiate ") + cudaGetErrorName(inst);
        }
        cudaGraphDestroy(graph);
    }
    const bool ok = pending == cudaSuccess && end == cudaSuccess;
    g_failures += ok ? 0 : 1;
    printf("%-44s %s | %s | pending %s | end %s | nodes %zu (kernel %d, mem-alloc %d, mem-free %d) | %s\n", name,
           ok ? "OK  " : "FAIL", calls.c_str(), cudaGetErrorName(pending), cudaGetErrorName(end), c.total, c.kernels,
           c.allocs, c.frees, run.c_str());
    (void) cudaGetLastError();
}

struct gemm_case {
    const char *       name;
    int                m, n, k;
    cublasMath_t       math;
    size_t             workspace;
    bool               gemm_ex;       // cublasGemmEx with compute_type instead of cublasSgemm
    cublasComputeType_t compute_type;
    bool               device_scalars;  // alpha/beta in device memory (CUBLAS_POINTER_MODE_DEVICE)
};

static void run_suite(const char * label, cudaStream_t stream, cublasHandle_t handle) {
    printf("\n=== %s ===\n", label);

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("cudaMemGetInfo: %zu MiB free of %zu MiB\n", free_b >> 20, total_b >> 20);

    // 1. Plain stream-ordered allocation, eager then captured. No cuBLAS.
    {
        void *            p = nullptr;
        const cudaError_t a = cudaMallocAsync(&p, 1 << 20, stream);
        const cudaError_t f = p ? cudaFreeAsync(p, stream) : cudaSuccess;
        const cudaError_t s = cudaStreamSynchronize(stream);
        printf("%-44s %s | malloc %s, free %s, sync %s\n", "eager cudaMallocAsync 1 MiB",
               a == cudaSuccess ? "OK  " : "FAIL", cudaGetErrorName(a), cudaGetErrorName(f), cudaGetErrorName(s));
        (void) cudaGetLastError();
    }
    for (size_t size : { size_t(4) << 10, size_t(1) << 20, size_t(64) << 20 }) {
        char name[64];
        snprintf(name, sizeof(name), "captured cudaMallocAsync %zu KiB", size >> 10);
        captured(name, stream, [&]() {
            void *            p = nullptr;
            const cudaError_t a = cudaMallocAsync(&p, size, stream);
            const cudaError_t f = p ? cudaFreeAsync(p, stream) : cudaSuccess;
            return std::string("malloc ") + cudaGetErrorName(a) + ", free " + cudaGetErrorName(f);
        });
    }

    // 2. The LoRA-A matmul as ggml issues it, and variants.
    const int  max_k  = 6144, max_mn = 2048;
    float *    A = nullptr, *B = nullptr, *C = nullptr, *dev_scalars = nullptr;
    void *     workspace = nullptr;
    const size_t max_ws = size_t(64) << 20;
    cudaMalloc(&A, sizeof(float) * max_k * max_mn);
    cudaMalloc(&B, sizeof(float) * max_k * 128);
    cudaMalloc(&C, sizeof(float) * max_mn * 128);
    cudaMalloc(&dev_scalars, 2 * sizeof(float));
    cudaMalloc(&workspace, max_ws);
    const float host_scalars[2] = { 1.0f, 0.0f };
    cudaMemcpy(dev_scalars, host_scalars, sizeof(host_scalars), cudaMemcpyHostToDevice);

    const gemm_case cases[] = {
        { "cublasSgemm 8x128x2048 TF32 ws4M (= ggml)", 8, 128, 2048, CUBLAS_TF32_TENSOR_OP_MATH, size_t(4) << 20, false, CUBLAS_COMPUTE_32F, false },
        { "cublasSgemm 8x128x6144 TF32 ws4M (= ggml)", 8, 128, 6144, CUBLAS_TF32_TENSOR_OP_MATH, size_t(4) << 20, false, CUBLAS_COMPUTE_32F, false },
        { "cublasSgemm 8x128x2048 default math",       8, 128, 2048, CUBLAS_DEFAULT_MATH, size_t(4) << 20, false, CUBLAS_COMPUTE_32F, false },
        { "cublasSgemm 8x128x2048 TF32 ws64M",          8, 128, 2048, CUBLAS_TF32_TENSOR_OP_MATH, max_ws, false, CUBLAS_COMPUTE_32F, false },
        { "cublasSgemm 8x128x2048 TF32 device scalars", 8, 128, 2048, CUBLAS_TF32_TENSOR_OP_MATH, size_t(4) << 20, false, CUBLAS_COMPUTE_32F, true },
        { "cublasGemmEx 8x128x2048 COMPUTE_32F",        8, 128, 2048, CUBLAS_DEFAULT_MATH, size_t(4) << 20, true, CUBLAS_COMPUTE_32F, false },
        { "cublasGemmEx 8x128x2048 32F_FAST_TF32",      8, 128, 2048, CUBLAS_DEFAULT_MATH, size_t(4) << 20, true, CUBLAS_COMPUTE_32F_FAST_TF32, false },
        { "cublasSgemm 2048x128x8 TF32 (LoRA-B shape)", 2048, 128, 8, CUBLAS_TF32_TENSOR_OP_MATH, size_t(4) << 20, false, CUBLAS_COMPUTE_32F, false },
    };

    for (const gemm_case & g : cases) {
        cublasSetMathMode(handle, g.math);
        cublasSetWorkspace(handle, workspace, g.workspace);
        cublasSetPointerMode(handle, g.device_scalars ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST);
        const float * alpha = g.device_scalars ? dev_scalars : &host_scalars[0];
        const float * beta  = g.device_scalars ? dev_scalars + 1 : &host_scalars[1];
        auto gemm = [&]() {
            if (g.gemm_ex) {
                return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, g.m, g.n, g.k, alpha, A, CUDA_R_32F, g.k, B,
                                    CUDA_R_32F, g.k, beta, C, CUDA_R_32F, g.m, g.compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            }
            return cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, g.m, g.n, g.k, alpha, A, g.k, B, g.k, beta, C, g.m);
        };
        // ggml runs the step eagerly before capturing it; do the same.
        const cublasStatus_t warm = gemm();
        cudaStreamSynchronize(stream);
        (void) cudaGetLastError();
        if (warm != CUBLAS_STATUS_SUCCESS) {
            printf("%-44s eager call failed: %s\n", g.name, cublas_status(warm));
        }
        captured(g.name, stream, [&]() { return std::string("cublas ") + cublas_status(gemm()); });
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    // Handles without a user workspace: one that never had cublasSetWorkspace, and one cleared with
    // (nullptr, 0). If these allocate during capture, ggml's calls are not using its 4 MiB workspace.
    for (int cleared = 0; cleared < 2; cleared++) {
        cublasHandle_t h2;
        cublasCreate(&h2);
        cublasSetStream(h2, stream);
        cublasSetMathMode(h2, CUBLAS_TF32_TENSOR_OP_MATH);
        if (cleared) {
            cublasSetWorkspace(h2, nullptr, 0);
        }
        auto gemm = [&]() {
            return cublasSgemm(h2, CUBLAS_OP_T, CUBLAS_OP_N, 8, 128, 2048, &host_scalars[0], A, 2048, B, 2048,
                               &host_scalars[1], C, 8);
        };
        gemm();
        cudaStreamSynchronize(stream);
        (void) cudaGetLastError();
        captured(cleared ? "cublasSgemm 8x128x2048 workspace (null, 0)" : "cublasSgemm 8x128x2048 no workspace set",
                 stream, [&]() { return std::string("cublas ") + cublas_status(gemm()); });
        cublasDestroy(h2);
    }

    size_t reserved = 0, used = 0;
    cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrReservedMemHigh, &reserved);
    cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &used);
    printf("graph memory pool high water: reserved %zu KiB, used %zu KiB\n", reserved >> 10, used >> 10);
    (void) cudaGetLastError();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(dev_scalars);
    cudaFree(workspace);
}

int main() {
    int driver = 0, runtime = 0, mempools = 0, cublas_ver = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    cudaDeviceGetAttribute(&mempools, cudaDevAttrMemoryPoolsSupported, 0);
    printf("device: %s (sm_%d%d, %d SMs), driver API %d, runtime %d, memory pools supported: %d\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount, driver, runtime, mempools);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);  // ggml's stream flags
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    cublasGetVersion(handle, &cublas_ver);
    printf("cuBLAS %d\n", cublas_ver);

    run_suite("clean process", stream, handle);

    // ggml-like memory state: the VMM pool reserves 32 GiB of address space and maps physical memory
    // into it as it grows; model, KV cache and compute buffers come from cudaMalloc.
    CUdeviceptr reserved_va = 0;
    CUresult    r           = cuMemAddressReserve(&reserved_va, size_t(1) << 35, 0, 0, 0);
    printf("\ncuMemAddressReserve 32 GiB: %d\n", (int) r);
    CUmemGenericAllocationHandle phys = 0;
    size_t                       mapped = 0;
    if (r == CUDA_SUCCESS) {
        CUmemAllocationProp ap = {};
        ap.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        ap.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
        ap.location.id         = 0;
        size_t gran            = 0;
        cuMemGetAllocationGranularity(&gran, &ap, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        const size_t size = ((size_t(1) << 30) + gran - 1) / gran * gran;
        if (cuMemCreate(&phys, size, &ap, 0) == CUDA_SUCCESS && cuMemMap(reserved_va, size, 0, phys, 0) == CUDA_SUCCESS) {
            CUmemAccessDesc access = {};
            access.location        = ap.location;
            access.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(reserved_va, size, &access, 1);
            mapped = size;
        }
        printf("mapped into the reservation: %zu MiB\n", mapped >> 20);
    }
    std::vector<void *> blocks;
    for (int i = 0; i < 7; i++) {
        void * p = nullptr;
        if (cudaMalloc(&p, size_t(512) << 20) == cudaSuccess) {
            blocks.push_back(p);
        }
    }
    printf("cudaMalloc'd: %zu x 512 MiB\n", blocks.size());
    (void) cudaGetLastError();

    run_suite("after a ggml-like memory state", stream, handle);

    for (void * p : blocks) {
        cudaFree(p);
    }
    if (mapped) {
        cuMemUnmap(reserved_va, mapped);
        cuMemRelease(phys);
    }
    if (reserved_va) {
        cuMemAddressFree(reserved_va, size_t(1) << 35);
    }
    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    printf("\n%d capture test(s) failed\n", g_failures);
    return g_failures ? 1 : 0;
}
