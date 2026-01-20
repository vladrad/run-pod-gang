"""
Example kernel using reference implementation.
Replace this with your optimized CUDA kernel.
"""
import torch

sf_vec_size = 16

def ceil_div(a, b):
    return (a + b - 1) // b

def to_blocked(input_matrix):
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


def custom_kernel(data):
    """
    NVFP4 block-scaled group GEMM.

    Input:
        data = (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
        - abc_tensors: list of (a, b, c) per group
            a: [m, k//2, l] float4_e2m1fn_x2
            b: [n, k//2, l] float4_e2m1fn_x2
            c: [m, n, l] float16 (output buffer)
        - sfasfb_tensors: list of (sfa, sfb) per group
            sfa: [m, k//16, l] float8_e4m3fn
            sfb: [n, k//16, l] float8_e4m3fn
        - sfasfb_reordered_tensors: list of (sfa_reordered, sfb_reordered) per group
            Reordered layout for cuBLAS
        - problem_sizes: list of (m, n, k, l) per group

    Output:
        list of c tensors (one per group)
    """
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []
    for (a_ref, b_ref, c_ref), (sfa_ref, sfb_ref), (m, n, k, l) in zip(
        abc_tensors, sfasfb_tensors, problem_sizes
    ):
        for l_idx in range(l):
            scale_a = to_blocked(sfa_ref[:, :, l_idx])
            scale_b = to_blocked(sfb_ref[:, :, l_idx])

            # C = A @ B.T with block scaling
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
