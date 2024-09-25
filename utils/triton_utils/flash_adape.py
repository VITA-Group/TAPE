"""
*Experimental* implementation of FlashAttention in Triton.
Tested with triton==2.0.0.dev20221202.
Triton 2.0 has a new backend (MLIR) but seems like it doesn't yet work for head dimensions
other than 64:
https://github.com/openai/triton/blob/d376020f90002757eea3ea9475d4f7cfc2ec5ead/python/triton/ops/flash_attention.py#L207
We'll update this implementation with the new Triton backend once this is fixed.

We use the FlashAttention implementation from Phil Tillet a starting point.
https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py

Changes:
- Implement both causal and non-causal attention.
- Implement both self-attention and cross-attention.
- Support arbitrary seqlens (not just multiples of 128), for both forward and backward.
- Support all head dimensions up to 128 (not just 16, 32, 64, 128), for both forward and backward.
- Support attention bias.
- Speed up the forward pass a bit, and only store the LSE instead of m and l.
- Make the backward for d=128 much faster by reducing register spilling.
- Optionally parallelize the backward pass across seqlen_k, to deal with the case of
small batch size * nheads.

Caution:
- This is an *experimental* implementation. The forward pass should be quite robust but
I'm not 100% sure that the backward pass doesn't have race conditions (due to the Triton compiler).
- This implementation has only been tested on A100.
- If you plan to use headdim other than 64 and 128, you should test for race conditions
(due to the Triton compiler), as done in tests/test_flash_attn.py
"test_flash_attn_triton_race_condition". I've tested and fixed many race conditions
for different head dimensions (40, 48, 64, 128, 80, 88, 96), but I'm still not 100% confident
that there are none left for other head dimensions.

Differences between this Triton version and the CUDA version:
- Triton version doesn't support dropout.
- Triton forward is generally faster than CUDA forward, while Triton backward is
generally slower than CUDA backward. Overall Triton forward + backward is slightly slower
than CUDA forward + backward.
- Triton version doesn't support different sequence lengths in a batch (i.e., RaggedTensor/NestedTensor).
- Triton version supports attention bias, while CUDA version doesn't.
"""

import copy
import math

import torch
import triton
import triton.language as tl
from .flash_attn import _flash_attn_backward as old_flash_attn_backward


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    V0,
    V1,
    Bias,
    Out,
    Out0,
    Out1,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_v0b,
    stride_v0h,
    stride_v0n,
    stride_v1b,
    stride_v1h,
    stride_v1n,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    stride_o0b,
    stride_o0h,
    stride_o0m,
    stride_o1b,
    stride_o1h,
    stride_o1m,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # off_b = tl.program_id(1)
    # off_h = tl.program_id(2)
    # off_hb = off_b * nheads + off_h
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # Initialize pointers to Q, K, V
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    v0_ptrs = (
        V0 + off_b * stride_v0b + off_h * stride_v0h + (offs_n[:, None] * stride_v0n + offs_d[None, :])
    )
    v1_ptrs = (
        V1 + off_b * stride_v1b + off_h * stride_v1h + (offs_n[:, None] * stride_v1n + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o0 = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    acc_o1 = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # scale acc_o
        acc_o_scale = tl.exp(m_i - m_ij)

        # # -- update output accumulator --
        # BUG: have to store and immediately load
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]
        acc_o0 = acc_o0 * acc_o_scale[:, None]
        acc_o1 = acc_o1 * acc_o_scale[:, None]

        # update acc_o
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
                v0 = tl.load(v0_ptrs + start_n * stride_v0n)
                v1 = tl.load(v1_ptrs + start_n * stride_v1n)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
                v0 = tl.load(v0_ptrs + start_n * stride_v0n, mask=offs_d[None, :] < headdim, other=0.0)
                v1 = tl.load(v1_ptrs + start_n * stride_v1n, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                v0 = tl.load(
                    v0_ptrs + start_n * stride_v0n,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
                v1 = tl.load(
                    v1_ptrs + start_n * stride_v1n,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                v0 = tl.load(
                    v0_ptrs + start_n * stride_v0n,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
                v1 = tl.load(
                    v1_ptrs + start_n * stride_v1n,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)

        acc_o += tl.dot(p, v)
        acc_o0 += tl.dot(p, v0)
        acc_o1 += tl.dot(p, v1)

        # -- update statistics
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    acc_o0 = acc_o0 * o_scale[:, None]
    acc_o1 = acc_o1 * o_scale[:, None]
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    # initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    out0_ptrs = (
        Out0
        + off_b * stride_o0b
        + off_h * stride_o0h
        + (offs_m[:, None] * stride_o0m + offs_d[None, :])
    )
    out1_ptrs = (
        Out1
        + off_b * stride_o1b
        + off_h * stride_o1h
        + (offs_m[:, None] * stride_o1m + offs_d[None, :])
    )

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
            tl.store(out0_ptrs, acc_o0)
            tl.store(out1_ptrs, acc_o1)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
            tl.store(out0_ptrs, acc_o0, mask=offs_d[None, :] < headdim)
            tl.store(out1_ptrs, acc_o1, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
            tl.store(out0_ptrs, acc_o0, mask=offs_m[:, None] < seqlen_q)
            tl.store(out1_ptrs, acc_o1, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )
            tl.store(
                out0_ptrs, acc_o0, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )
            tl.store(
                out1_ptrs, acc_o1, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    Out0,
    Out1,
    DO,
    DO0,
    DO1,
    Delta,
    Delta0,
    Delta1,
    stride_ob,
    stride_oh,
    stride_om,
    stride_o0b,
    stride_o0h,
    stride_o0m,
    stride_o1b,
    stride_o1h,
    stride_o1m,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_do0b,
    stride_do0h,
    stride_do0m,
    stride_do1b,
    stride_do1h,
    stride_do1m,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    o0 = tl.load(
        Out0 + off_b * stride_o0b + off_h * stride_o0h + offs_m[:, None] * stride_o0m + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    o1 = tl.load(
        Out1 + off_b * stride_o1b + off_h * stride_o1h + offs_m[:, None] * stride_o1m + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do0 = tl.load(
        DO0
        + off_b * stride_do0b
        + off_h * stride_do0h
        + offs_m[:, None] * stride_do0m
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do1 = tl.load(
        DO1
        + off_b * stride_do1b
        + off_h * stride_do1h
        + offs_m[:, None] * stride_do1m
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    delta0 = tl.sum(o0 * do0, axis=1)
    delta1 = tl.sum(o1 * do1, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)
    tl.store(Delta0 + off_hb * seqlen_q_rounded + offs_m, delta0)
    tl.store(Delta1 + off_hb * seqlen_q_rounded + offs_m, delta1)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dk0_ptrs,
    dk1_ptrs,
    dv_ptrs,
    dv0_ptrs,
    dv1_ptrs,
    dk,
    dk0,
    dk1,
    dv,
    dv0,
    dv1,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dv0_ptrs, dv0)
            tl.store(dv1_ptrs, dv1)
            tl.store(dk_ptrs, dk)
            tl.store(dk0_ptrs, dk0)
            tl.store(dk1_ptrs, dk1)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dv0_ptrs, dv0, mask=offs_d[None, :] < headdim)
            tl.store(dv1_ptrs, dv1, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            tl.store(dk0_ptrs, dk0, mask=offs_d[None, :] < headdim)
            tl.store(dk1_ptrs, dk1, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dv0_ptrs, dv0, mask=offs_n[:, None] < seqlen_k)
            tl.store(dv1_ptrs, dv1, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk0_ptrs, dk0, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk1_ptrs, dk1, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dv0_ptrs, dv0, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dv1_ptrs, dv1, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk0_ptrs, dk0, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk1_ptrs, dk1, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    V0,
    V1,
    Bias,
    DO,
    DO0,
    DO1,
    DQ,
    DQ0,
    DQ1,
    DK,
    DK0,
    DK1,
    DV,
    DV0,
    DV1,
    LSE,
    D,
    D0,
    D1,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_v0n,
    stride_v1n,
    stride_bm,
    stride_dom,
    stride_do0m,
    stride_do1m,
    stride_dqm,
    stride_dq0m,
    stride_dq1m,
    stride_dkn,
    stride_dk0n,
    stride_dk1n,
    stride_dvn,
    stride_dv0n,
    stride_dv1n,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    v0_ptrs = V0 + (offs_n[:, None] * stride_v0n + offs_d[None, :])
    v1_ptrs = V1 + (offs_n[:, None] * stride_v1n + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    do0_ptrs = DO0 + (offs_qm[:, None] * stride_do0m + offs_d[None, :])
    do1_ptrs = DO1 + (offs_qm[:, None] * stride_do1m + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    dq0_ptrs = DQ0 + (offs_qm[:, None] * stride_dq0m + offs_d[None, :])
    dq1_ptrs = DQ1 + (offs_qm[:, None] * stride_dq1m + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv0 = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv1 = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk0 = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk1 = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dv0_ptrs = DV0 + (offs_n[:, None] * stride_dv0n + offs_d[None, :])
        dv1_ptrs = DV1 + (offs_n[:, None] * stride_dv1n + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        dk0_ptrs = DK0 + (offs_n[:, None] * stride_dk0n + offs_d[None, :])
        dk1_ptrs = DK1 + (offs_n[:, None] * stride_dk1n + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dk0_ptrs,
            dk1_ptrs,
            dv_ptrs,
            dv0_ptrs,
            dv1_ptrs,
            dk,
            dk0,
            dk1,
            dv,
            dv0,
            dv1,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
            v0 = tl.load(v0_ptrs)
            v1 = tl.load(v1_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v0 = tl.load(v0_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v1 = tl.load(v1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v0 = tl.load(v0_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v1 = tl.load(v1_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v0 = tl.load(
                v0_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v1 = tl.load(
                v1_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
            do0 = tl.load(do0_ptrs)
            do1 = tl.load(do1_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            do0 = tl.load(
                do0_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
            do1 = tl.load(
                do1_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dv0 += tl.dot(tl.trans(p.to(do0.dtype)), do0)
        dv1 += tl.dot(tl.trans(p.to(do1.dtype)), do1)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        dp = tl.dot(do, tl.trans(v))
        dp0 = tl.dot(do0, tl.trans(v0))
        dp1 = tl.dot(do1, tl.trans(v1))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        D0i = tl.load(D0 + offs_m_curr)
        D1i = tl.load(D1 + offs_m_curr)
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        ds0 = (p * (dp0 - D0i[:, None]) * softmax_scale).to(q.dtype)
        ds1 = (p * (dp1 - D1i[:, None]) * softmax_scale).to(q.dtype)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        dk0 += tl.dot(tl.trans(ds0), q)
        dk1 += tl.dot(tl.trans(ds1), q)
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq0 = tl.load(dq0_ptrs, eviction_policy="evict_last")
                dq1 = tl.load(dq1_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                dq0 += tl.dot(ds0, k)
                dq1 += tl.dot(ds1, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
                tl.store(dq0_ptrs, dq0, eviction_policy="evict_last")
                tl.store(dq1_ptrs, dq1, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq0 = tl.load(
                        dq0_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq1 = tl.load(
                        dq1_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    dq0 += tl.dot(ds0, k)
                    dq1 += tl.dot(ds1, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                    tl.store(
                        dq0_ptrs,
                        dq0,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                    tl.store(
                        dq1_ptrs,
                        dq1,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq0 = tl.load(
                        dq0_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq1 = tl.load(
                        dq1_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    dq0 += tl.dot(ds0, k)
                    dq1 += tl.dot(ds1, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
                    tl.store(
                        dq0_ptrs,
                        dq0,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
                    tl.store(
                        dq1_ptrs,
                        dq1,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            dq0 = tl.dot(ds0, k)
            dq1 = tl.dot(ds1, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
                tl.atomic_add(dq0_ptrs, dq0)
                tl.atomic_add(dq1_ptrs, dq1)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                    tl.atomic_add(dq0_ptrs, dq0, mask=offs_m_curr[:, None] < seqlen_q)
                    tl.atomic_add(dq1_ptrs, dq1, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
                    tl.atomic_add(
                        dq0_ptrs,
                        dq0,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
                    tl.atomic_add(
                        dq1_ptrs,
                        dq1,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        dq0_ptrs += BLOCK_M * stride_dq0m
        dq1_ptrs += BLOCK_M * stride_dq1m
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        do0_ptrs += BLOCK_M * stride_do0m
        do1_ptrs += BLOCK_M * stride_do1m
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dv0_ptrs = DV0 + (offs_n[:, None] * stride_dv0n + offs_d[None, :])
    dv1_ptrs = DV1 + (offs_n[:, None] * stride_dv1n + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    dk0_ptrs = DK0 + (offs_n[:, None] * stride_dk0n + offs_d[None, :])
    dk1_ptrs = DK1 + (offs_n[:, None] * stride_dk1n + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dk0_ptrs,
        dk1_ptrs,
        dv_ptrs,
        dv0_ptrs,
        dv1_ptrs,
        dk,
        dk0,
        dk1,
        dv,
        dv0,
        dv1,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    if isinstance(name, list):
        def func(nargs):
            for name_i in name:
                nargs[name_i].zero_()
        return func
    return lambda nargs: nargs[name].zero_()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DQ0", "DQ1"]),
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DQ0", "DQ1"]),
        ),       
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DQ0", "DQ1"]),
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero(["DQ", "DQ0", "DQ1"]),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    V0,
    V1,
    Bias,
    DO,
    DO0,
    DO1,
    DQ,
    DQ0,
    DQ1,
    DK,
    DK0,
    DK1,
    DV,
    DV0,
    DV1,
    LSE,
    D,
    D0,
    D1,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_v0b,
    stride_v0h,
    stride_v0n,
    stride_v1b,
    stride_v1h,
    stride_v1n,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_do0b,
    stride_do0h,
    stride_do0m,
    stride_do1b,
    stride_do1h,
    stride_do1m,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dq0b,
    stride_dq0h,
    stride_dq0m,
    stride_dq1b,
    stride_dq1h,
    stride_dq1m,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dk0b,
    stride_dk0h,
    stride_dk0n,
    stride_dk1b,
    stride_dk1h,
    stride_dk1n,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dv0b,
    stride_dv0h,
    stride_dv0n,
    stride_dv1b,
    stride_dv1h,
    stride_dv1n,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    V0 += off_b * stride_v0b + off_h * stride_v0h
    V1 += off_b * stride_v1b + off_h * stride_v1h
    DO += off_b * stride_dob + off_h * stride_doh
    DO0 += off_b * stride_do0b + off_h * stride_do0h
    DO1 += off_b * stride_do1b + off_h * stride_do1h
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DQ0 += off_b * stride_dq0b + off_h * stride_dq0h
    DQ1 += off_b * stride_dq1b + off_h * stride_dq1h
    DK += off_b * stride_dkb + off_h * stride_dkh
    DK0 += off_b * stride_dk0b + off_h * stride_dk0h
    DK1 += off_b * stride_dk1b + off_h * stride_dk1h
    DV += off_b * stride_dvb + off_h * stride_dvh
    DV0 += off_b * stride_dv0b + off_h * stride_dv0h
    DV1 += off_b * stride_dv1b + off_h * stride_dv1h
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    D0 += off_hb * seqlen_q_rounded
    D1 += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                V0,
                V1,
                Bias,
                DO,
                DO0,
                DO1,
                DQ,
                DQ0,
                DQ1,
                DK,
                DK0,
                DK1,
                DV,
                DV0,
                DV1,
                LSE,
                D,
                D0,
                D1,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_v0n,
                stride_v1n,
                stride_bm,
                stride_dom,
                stride_do0m,
                stride_do1m,
                stride_dqm,
                stride_dq0m,
                stride_dq1m,
                stride_dkn,
                stride_dk0n,
                stride_dk1n,
                stride_dvn,
                stride_dv0n,
                stride_dv1n,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            V0,
            V1,
            Bias,
            DO,
            DO0,
            DO1,
            DQ,
            DQ0,
            DQ1,
            DK,
            DK0,
            DK1,
            DV,
            DV0,
            DV1,
            LSE,
            D,
            D0,
            D1,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_v0n,
            stride_v1n,
            stride_bm,
            stride_dom,
            stride_do0m,
            stride_do1m,
            stride_dqm,
            stride_dq0m,
            stride_dq1m,
            stride_dkn,
            stride_dk0n,
            stride_dk1n,
            stride_dvn,
            stride_dv0n,
            stride_dv1n,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )


def _flash_attn_forward(q, k, v, v0, v1, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    o0 = torch.empty_like(q)
    o1 = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 2 if d <= 64 else 4
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _fwd_kernel[grid](
        q,
        k,
        v,
        v0,
        v1,
        bias,
        o,
        o0,
        o1,
        lse,
        tmp,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v0.stride(0),
        v0.stride(2),
        v0.stride(1),
        v1.stride(0),
        v1.stride(2),
        v1.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        o0.stride(0),
        o0.stride(2),
        o0.stride(1),
        o1.stride(0),
        o1.stride(2),
        o1.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, o0, o1, lse, softmax_scale  # softmax_scale could have been updated


def _flash_attn_backward(
    do, do0, do1, q, k, v, v0, v1, o, o0, o1, lse, dq, dq0, dq1, dk, dk0, dk1, dv, dv0, dv1, bias=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
        do0 = do0.contiguous()
        do1 = do1.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.empty_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    dq0_accum = torch.empty_like(q, dtype=torch.float32)
    dq1_accum = torch.empty_like(q, dtype=torch.float32)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    # dq0_accum = torch.zeros_like(q, dtype=torch.float32)
    # dq1_accum = torch.zeros_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    delta0 = torch.empty_like(lse)
    delta1 = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        o0,
        o1,
        do,
        do0,
        do1,
        delta,
        delta0,
        delta1,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        o0.stride(0),
        o0.stride(2),
        o0.stride(1),
        o1.stride(0),
        o1.stride(2),
        o1.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        do0.stride(0),
        do0.stride(2),
        do0.stride(1),
        do1.stride(0),
        do1.stride(2),
        do1.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=64,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    # BLOCK_M = 128
    # BLOCK_N = 64
    # BLOCK = 64
    # num_warps = 2
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        v0,
        v1,
        bias,
        do,
        do0,
        do1,
        dq_accum,
        dq0_accum,
        dq1_accum,
        dk,
        dk0,
        dk1,
        dv,
        dv0,
        dv1,
        lse,
        delta,
        delta0,
        delta1,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        v0.stride(0),
        v0.stride(2),
        v0.stride(1),
        v1.stride(0),
        v1.stride(2),
        v1.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        do0.stride(0),
        do0.stride(2),
        do0.stride(1),
        do1.stride(0),
        do1.stride(2),
        do1.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dq0_accum.stride(0),
        dq0_accum.stride(2),
        dq0_accum.stride(1),
        dq1_accum.stride(0),
        dq1_accum.stride(2),
        dq1_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dk0.stride(0),
        dk0.stride(2),
        dk0.stride(1),
        dk1.stride(0),
        dk1.stride(2),
        dk1.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        dv0.stride(0),
        dv0.stride(2),
        dv0.stride(1),
        dv1.stride(0),
        dv1.stride(2),
        dv1.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=True,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=8,
        # num_stages=1,
    )
    dq.copy_(dq_accum)
    dq0.copy_(dq0_accum)
    dq1.copy_(dq1_accum)


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, causal=False, softmax_scale=None, bias=None):
        """
        qkv: (batch, seqlen, 3, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen, seqlen).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen, seqlen)
        """
        # Make sure that the last dimension is contiguous
        if qkv.stride(-1) != 1:
            qkv = qkv.contiguous()
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            bias=bias,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(qkv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[1], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dqkv = torch.empty_like(qkv)
            _flash_attn_backward(
                do,
                qkv[:, :, 0],
                qkv[:, :, 1],
                qkv[:, :, 2],
                o,
                lse,
                dqkv[:, :, 0],
                dqkv[:, :, 1],
                dqkv[:, :, 2],
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dqkv, None, None, None


flash_attn_qkvpacked_func = FlashAttnQKVPackedFunc.apply


class FlashAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, bias=None, causal=False, softmax_scale=None):
        """
        q: (batch, seqlen_q, nheads, headdim)
        kv: (batch, seqlen_k, 2, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, kv = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, kv]]
        o, lse, ctx.softmax_scale = _flash_attn_forward(
            q, kv[:, :, 0], kv[:, :, 1], bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, kv, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o, lse, bias = ctx.saved_tensors
        if len(ctx.needs_input_grad) >= 3:
            assert not ctx.needs_input_grad[2], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dkv = torch.empty_like(kv)
            _flash_attn_backward(
                do,
                q,
                kv[:, :, 0],
                kv[:, :, 1],
                o,
                lse,
                dq,
                dkv[:, :, 0],
                dkv[:, :, 1],
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dkv, None, None, None


flash_attn_kvpacked_func = FlashAttnKVPackedFunc.apply


class AdaFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, v0, v1, causal, softmax_scale, bias=None):
        """
        q: (batch_size, seqlen_q, nheads, headdim)
        k, v: (batch_size, seqlen_k, nheads, headdim)
        bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
            For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
            ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v, v0, v1 = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v, v0, v1]]
        # from .flash_attn import _flash_attn_forward as old_flash_attn_forward
        # o, lse, ctx.softmax_scale = old_flash_attn_forward(
        #     q, k, v, bias=bias, causal=causal, softmax_scale=softmax_scale
        # ) 
        # o0, lse, ctx.softmax_scale = old_flash_attn_forward(
        #     q, k, v0, bias=bias, causal=causal, softmax_scale=softmax_scale
        # )
        # o1, lse, ctx.softmax_scale = old_flash_attn_forward(
        #     q, k, v1, bias=bias, causal=causal, softmax_scale=softmax_scale
        # )
        o, o0, o1, lse, ctx.softmax_scale = _flash_attn_forward(
            q, k, v, v0, v1, bias=bias, causal=causal, softmax_scale=softmax_scale
        )
        ctx.save_for_backward(q, k, v, v0, v1, o, o0, o1, lse, bias)
        ctx.causal = causal
        return o, o0, o1

    @staticmethod
    def backward(ctx, do, do0, do1):
        q, k, v, v0, v1, o, o0, o1, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[-1], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        # with torch.inference_mode():
        dq = torch.empty_like(q)
        dq0 = torch.empty_like(q)
        dq1 = torch.empty_like(q)
        dk = torch.empty_like(k)
        dk0 = torch.empty_like(k)
        dk1 = torch.empty_like(k)
        dv = torch.empty_like(v)
        dv0 = torch.empty_like(v)
        dv1 = torch.empty_like(v)
        old_flash_attn_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            bias=bias,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        )
        old_flash_attn_backward(
            do0,
            q,
            k,
            v0,
            o0,
            lse,
            dq0,
            dk0,
            dv0,
            bias=bias,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        )
        old_flash_attn_backward(
            do1,
            q,
            k,
            v1,
            o1,
            lse,
            dq1,
            dk1,
            dv1,
            bias=bias,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        )
        # _flash_attn_backward(
        #     do,
        #     do0,
        #     do1,
        #     q,
        #     k,
        #     v,
        #     v0,
        #     v1,
        #     o,
        #     o0,
        #     o1,
        #     lse,
        #     dq,
        #     dq0,
        #     dq1,
        #     dk,
        #     dk0,
        #     dk1,
        #     dv,
        #     dv0,
        #     dv1,
        #     bias=bias,
        #     causal=ctx.causal,
        #     softmax_scale=ctx.softmax_scale,
        # )
        dq += dq0 + dq1
        dk += dk0 + dk1
            # breakpoint()
        return dq, dk, dv, dv0, dv1, None, None, None


adape_flash_attn_func = AdaFlashAttnFunc.apply