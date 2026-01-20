"""
RunPod Serverless Handler for NVFP4 Block-Scaled Group GEMM on B200
Matches GPU Mode competition format exactly.
"""
import runpod
import torch
import time
import traceback
import math

# Scaling factor vector size
sf_vec_size = 16


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    """Convert scale factor tensor to blocked format for torch._scaled_mm."""
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
        padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


def create_reordered_scale_factor_tensor(l, mn, k, ref_f8_tensor):
    """
    Prepare scale factor tensors in the cuBLAS block-scaling layout.
    See: https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4#d-block-scaling-factors-layout
    """
    sf_k = ceil_div(k, sf_vec_size)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    rand_int_tensor = torch.randint(1, 3, mma_shape, dtype=torch.int8, device='cuda')
    reordered_f8_tensor = rand_int_tensor.to(dtype=torch.float8_e4m3fn)
    reordered_f8_tensor = reordered_f8_tensor.permute(*mma_permute_order)

    if ref_f8_tensor.device.type == 'cpu':
        ref_f8_tensor = ref_f8_tensor.cuda()

    i_idx = torch.arange(mn, device='cuda')
    j_idx = torch.arange(sf_k, device='cuda')
    b_idx = torch.arange(l, device='cuda')

    i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

    mm = i_grid // (atom_m[0] * atom_m[1])
    mm32 = i_grid % atom_m[0]
    mm4 = (i_grid % 128) // atom_m[0]
    kk = j_grid // atom_k
    kk4 = j_grid % atom_k

    reordered_f8_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_tensor[i_grid, j_grid, b_grid]

    return reordered_f8_tensor


def generate_input(m: tuple, n: tuple, k: tuple, g: int, seed: int):
    """
    Generate input tensors for NVFP4 block-scaled group GEMM.

    Returns:
        Tuple of (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
        - abc_tensors: list of (a, b, c) per group
        - sfasfb_tensors: list of (sfa, sfb) per group
        - sfasfb_reordered_tensors: list of (sfa_reordered, sfb_reordered) per group
        - problem_sizes: list of (m, n, k, l) per group
    """
    torch.manual_seed(seed)

    abc_tensors = []
    sfasfb_tensors = []
    sfasfb_reordered_tensors = []
    problem_sizes = []
    l = 1

    for group_idx in range(g):
        mi = m[group_idx]
        ni = n[group_idx]
        ki = k[group_idx]

        # A matrix: [m, k//2, l] as int8 -> view as float4_e2m1fn_x2
        a_ref = torch.randint(
            -1, 2, (l, mi, ki // 2), dtype=torch.int8, device="cuda"
        ).permute(1, 2, 0)
        a_ref = a_ref.view(torch.float4_e2m1fn_x2)

        # B matrix: [n, k//2, l] as int8 -> view as float4_e2m1fn_x2
        b_ref = torch.randint(
            -1, 2, (l, ni, ki // 2), dtype=torch.int8, device="cuda"
        ).permute(1, 2, 0)
        b_ref = b_ref.view(torch.float4_e2m1fn_x2)

        # C matrix: [m, n, l] as float16
        c_ref = torch.randn((l, mi, ni), dtype=torch.float16, device="cuda").permute(1, 2, 0)

        # Scale factors
        sf_k = ceil_div(ki, sf_vec_size)
        sfa_ref_cpu = torch.randint(
            1, 3, (l, mi, sf_k), dtype=torch.int8
        ).to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)
        sfb_ref_cpu = torch.randint(
            1, 3, (l, ni, sf_k), dtype=torch.int8
        ).to(dtype=torch.float8_e4m3fn).permute(1, 2, 0)

        # Reordered scale factors for custom kernels
        sfa_reordered = create_reordered_scale_factor_tensor(l, mi, ki, sfa_ref_cpu)
        sfb_reordered = create_reordered_scale_factor_tensor(l, ni, ki, sfb_ref_cpu)

        abc_tensors.append((a_ref, b_ref, c_ref))
        sfasfb_tensors.append((sfa_ref_cpu, sfb_ref_cpu))
        sfasfb_reordered_tensors.append((sfa_reordered, sfb_reordered))
        problem_sizes.append((mi, ni, ki, l))

    return (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)


def ref_kernel(data):
    """Reference implementation using torch._scaled_mm."""
    abc_tensors, sfasfb_tensors, _, problem_sizes = data

    result_tensors = []
    for (a_ref, b_ref, c_ref), (sfa_ref, sfb_ref), (m, n, k, l) in zip(
        abc_tensors, sfasfb_tensors, problem_sizes
    ):
        for l_idx in range(l):
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])
            res = torch._scaled_mm(
                a_ref[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b_ref[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a.cuda(),
                scale_b.cuda(),
                bias=None,
                out_dtype=torch.float16,
            )
            c_ref[:, :, l_idx] = res
        result_tensors.append(c_ref)
    return result_tensors


def check_correctness(custom_results, ref_results, rtol=1e-3, atol=1e-3):
    """Check if custom kernel output matches reference."""
    for i, (custom, ref) in enumerate(zip(custom_results, ref_results)):
        if not torch.allclose(custom, ref, rtol=rtol, atol=atol):
            max_diff = (custom - ref).abs().max().item()
            return False, f"Group {i} mismatch, max_diff={max_diff}"
    return True, "OK"


def benchmark_kernel(kernel_fn, data, warmup=5, runs=20):
    """Benchmark kernel with CUDA events."""
    # Warmup
    for _ in range(warmup):
        kernel_fn(data)

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(runs):
        start.record()
        kernel_fn(data)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }


def handler(event):
    """
    Input format:
    {
        "kernel_code": "def custom_kernel(data): ...",
        "test_cases": [
            {"m": [128, 256], "n": [128, 256], "k": [512, 512], "g": 2},
            ...
        ],
        "warmup": 5,
        "runs": 20,
        "check_correctness": true,
        "seed": 42
    }

    The custom_kernel function receives:
        data = (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)

    And should return:
        list of output tensors (one per group)
    """
    try:
        input_data = event.get("input", {})
        kernel_code = input_data.get("kernel_code", "")
        test_cases = input_data.get("test_cases", [
            # Default test case: 3 groups with different sizes
            {"m": (7168, 4096, 7168), "n": (7168, 4096, 7168), "k": (16384, 7168, 2048), "g": 3},
        ])
        warmup = input_data.get("warmup", 5)
        runs = input_data.get("runs", 20)
        do_check = input_data.get("check_correctness", True)
        seed = input_data.get("seed", 42)

        if not kernel_code:
            return {"error": "No kernel_code provided"}

        # GPU info
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
        }

        # Load custom kernel
        exec_globals = {"torch": torch}
        exec(kernel_code, exec_globals)

        if "custom_kernel" not in exec_globals:
            return {"error": "Must define 'custom_kernel(data)' function"}

        custom_kernel = exec_globals["custom_kernel"]

        # Run benchmarks for each test case
        results = []
        for tc in test_cases:
            m = tuple(tc["m"])
            n = tuple(tc["n"])
            k = tuple(tc["k"])
            g = tc["g"]

            # Generate data
            data = generate_input(m, n, k, g, seed)

            # Check correctness if requested
            correctness = {"passed": True, "message": "skipped"}
            if do_check:
                # Clone data for reference (since kernels may modify in-place)
                ref_data = generate_input(m, n, k, g, seed)
                custom_data = generate_input(m, n, k, g, seed)

                ref_results = ref_kernel(ref_data)
                custom_results = custom_kernel(custom_data)

                passed, msg = check_correctness(custom_results, ref_results)
                correctness = {"passed": passed, "message": msg}

            # Benchmark
            timing = benchmark_kernel(custom_kernel, data, warmup, runs)

            results.append({
                "m": m,
                "n": n,
                "k": k,
                "g": g,
                "correctness": correctness,
                **timing
            })

        # Compute geometric mean of mean times
        geomean = math.exp(sum(math.log(r["mean_ms"]) for r in results) / len(results))

        return {
            "gpu": gpu_info,
            "results": results,
            "geomean_ms": geomean,
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
