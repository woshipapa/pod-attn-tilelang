
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple, Union

import torch

try:
    import tilelang
    import tilelang.language as T
    from tilelang.carver.arch import driver
except Exception:  # pragma: no cover
    tilelang = None
    T = None
    driver = None


def _require_tilelang() -> None:
    if tilelang is None or T is None or driver is None:
        raise RuntimeError("TileLang is required but not available in the current environment.")


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _num_splits_heuristic(batch_nheads_mblocks: int, num_sms: int, num_n_blocks: int, max_splits: int) -> int:
    if batch_nheads_mblocks >= int(0.8 * num_sms):
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits < 1:
        return 1

    def is_split_eligible(num_splits: int) -> bool:
        return num_splits == 1 or _ceildiv(num_n_blocks, num_splits) != _ceildiv(num_n_blocks, num_splits - 1)

    max_eff = 0.0
    effs = [0.0] * max_splits
    for s in range(1, max_splits + 1):
        if not is_split_eligible(s):
            continue
        n_waves = float(batch_nheads_mblocks * s) / float(num_sms)
        eff = n_waves / float(int(n_waves + 0.999999))
        effs[s - 1] = eff
        if eff > max_eff:
            max_eff = eff

    for s in range(1, max_splits + 1):
        if not is_split_eligible(s):
            continue
        if effs[s - 1] >= 0.85 * max_eff:
            return s
    return 1


def _resolve_split_counts(
    fused_params: int,
    batch_p: int,
    heads_p: int,
    seq_q_p: int,
    seq_k_p: int,
    batch_d: int,
    heads_d: int,
    seq_q_d: int,
    seq_k_d: int,
    head_dim: int,
    num_splits_p: int,
    num_splits_d: int,
    sm_count: int,
) -> Tuple[int, int]:
    num_m_blocks_p = _ceildiv(seq_q_p, 128)
    num_m_blocks_d = _ceildiv(seq_q_d, 64)
    prefill_work = batch_p * heads_p * num_m_blocks_p
    decode_work = batch_d * heads_d * num_m_blocks_d

    splits_p = int(num_splits_p)
    splits_d = int(num_splits_d)

    if fused_params == 9 or (
        fused_params == 15 and (prefill_work >= sm_count * 2 or prefill_work * 2 < int(0.8 * sm_count * 2))
    ):
        if int(0.8 * sm_count * 2) <= decode_work < sm_count * 2:
            splits_d = sm_count * 2 // decode_work + 1

    max_splits_p = 128 if ((fused_params & 8) == 0 or splits_p != 0) else max(1, (2 * sm_count * 2) // max(1, prefill_work * 2))

    if (fused_params & 8) and splits_p == 0 and decode_work >= sm_count * 2:
        splits_p = 1

    block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
    num_n_blocks_p = _ceildiv(seq_k_p, block_n)
    num_n_blocks_d = _ceildiv(seq_k_d, block_n)

    if splits_p < 1:
        splits_p = _num_splits_heuristic(prefill_work, sm_count * 2, num_n_blocks_p, max_splits=max_splits_p)
    if splits_d < 1:
        splits_d = _num_splits_heuristic(decode_work, sm_count * 2, num_n_blocks_d, max_splits=128)

    splits_p = max(1, min(128, splits_p))
    splits_d = max(1, min(128, splits_d))
    return splits_p, splits_d


def _choose_fused_op_auto(
    batch_p: int,
    heads_p: int,
    seq_q_p: int,
    seq_k_p: int,
    splits_p: int,
    batch_d: int,
    heads_d: int,
    seq_q_d: int,
    seq_k_d: int,
    splits_d: int,
    sm_count: int,
) -> int:
    dim_p = 64 if splits_p > 1 else 128
    prefill_work = batch_p * heads_p * _ceildiv(seq_q_p, dim_p) * splits_p
    decode_work = batch_d * heads_d * ((seq_q_d + 64) // 64) * splits_d

    if splits_p > 1 or splits_d > 1:
        return 9
    if prefill_work < sm_count * 2 and decode_work < sm_count * 3:
        return 11
    if prefill_work < sm_count * 2 and seq_k_p * 8 >= seq_k_d * 5:
        return 9
    if prefill_work >= sm_count * 2 and decode_work < sm_count * 2 and seq_k_p * 8 >= seq_k_d * 5:
        return 11
    if decode_work >= prefill_work:
        return 11
    return 9


def _select_tile_shapes(fused_op: int, decode_split: bool, prefill_split: bool) -> Tuple[int, int, int, int]:
    if fused_op & 8:
        block_m_d, block_n_d = (64, 128) if decode_split else (16, 32)
        if prefill_split:
            block_m_p, block_n_p = 64, 128
        elif fused_op & 2:
            block_m_p, block_n_p = 64, 32
        elif fused_op & 4:
            block_m_p, block_n_p = 16, 32
        else:
            block_m_p, block_n_p = 128, 64
    else:
        if fused_op & 2:
            block_m_p, block_n_p = 64, 32
            block_m_d, block_n_d = 32, 64
        elif fused_op & 4:
            block_m_p, block_n_p = 16, 32
            block_m_d, block_n_d = 16, 32
        else:
            block_m_p, block_n_p = 128, 64
            block_m_d, block_n_d = 64, 128
    return block_m_p, block_n_p, block_m_d, block_n_d


def get_tilelang_fused_launch_plan(
    q_p: torch.Tensor,
    k_cache_p: torch.Tensor,
    q_d: torch.Tensor,
    k_cache_d: torch.Tensor,
    fused_params: int = 15,
    num_splits_p: int = 0,
    num_splits_d: int = 0,
) -> dict:
    _require_tilelang()
    if fused_params not in (8, 9, 10, 11, 15, 64):
        raise ValueError("fused_params must be one of {8, 9, 10, 11, 15, 64}.")

    batch_p, seq_q_p, heads, dim = q_p.shape
    batch_d, seq_q_d, heads_d, dim_d = q_d.shape
    if heads != heads_d or dim != dim_d:
        raise ValueError("prefill and decode shapes mismatch.")

    sm_count = torch.cuda.get_device_properties(q_p.device).multi_processor_count
    splits_p, splits_d = _resolve_split_counts(
        fused_params=fused_params,
        batch_p=batch_p,
        heads_p=heads,
        seq_q_p=seq_q_p,
        seq_k_p=k_cache_p.shape[1],
        batch_d=batch_d,
        heads_d=heads_d,
        seq_q_d=seq_q_d,
        seq_k_d=k_cache_d.shape[1],
        head_dim=dim,
        num_splits_p=num_splits_p,
        num_splits_d=num_splits_d,
        sm_count=sm_count,
    )

    chosen_fused_op = fused_params
    if fused_params == 15:
        chosen_fused_op = _choose_fused_op_auto(
            batch_p=batch_p,
            heads_p=heads,
            seq_q_p=seq_q_p,
            seq_k_p=k_cache_p.shape[1],
            splits_p=splits_p,
            batch_d=batch_d,
            heads_d=heads_d,
            seq_q_d=seq_q_d,
            seq_k_d=k_cache_d.shape[1],
            splits_d=splits_d,
            sm_count=sm_count,
        )

    decode_split = splits_d > 1
    prefill_split = splits_p > 1
    block_m_p, block_n_p, block_m_d, block_n_d = _select_tile_shapes(chosen_fused_op, decode_split, prefill_split)

    q_tiles_p = _ceildiv(seq_q_p, block_m_p)
    q_tiles_d = _ceildiv(seq_q_d, block_m_d)
    prefill_blocks = q_tiles_p * batch_p * heads * splits_p
    decode_blocks = q_tiles_d * batch_d * heads * splits_d

    return {
        "sm_count": sm_count,
        "fused_params_input": fused_params,
        "fused_op_selected": chosen_fused_op,
        "num_splits_p": splits_p,
        "num_splits_d": splits_d,
        "prefill_split": prefill_split,
        "decode_split": decode_split,
        "block_m_p": block_m_p,
        "block_n_p": block_n_p,
        "block_m_d": block_m_d,
        "block_n_d": block_n_d,
        "threads": 128,
        "prefill_blocks": prefill_blocks,
        "decode_blocks": decode_blocks,
        "scheduler_counters": sm_count + 2,
    }


def _parse_cache_seqlens(
    cache_seqlens: Optional[Union[int, torch.Tensor]],
    batch: int,
    full_seqlen: int,
    device: torch.device,
) -> torch.Tensor:
    if cache_seqlens is None:
        return torch.full((batch,), full_seqlen, dtype=torch.int32, device=device)
    if isinstance(cache_seqlens, int):
        return torch.full((batch,), cache_seqlens, dtype=torch.int32, device=device)
    if isinstance(cache_seqlens, torch.Tensor):
        if cache_seqlens.numel() != batch:
            raise ValueError("cache_seqlens tensor size mismatch.")
        if cache_seqlens.dtype != torch.int32:
            cache_seqlens = cache_seqlens.to(dtype=torch.int32)
        return cache_seqlens.to(device=device)
    raise TypeError("cache_seqlens must be int / tensor / None.")


def _validate_inputs(
    q_p: torch.Tensor,
    k_cache_p: torch.Tensor,
    v_cache_p: torch.Tensor,
    q_d: torch.Tensor,
    k_cache_d: torch.Tensor,
    v_cache_d: torch.Tensor,
) -> None:
    for name, x in [
        ("q_p", q_p),
        ("k_cache_p", k_cache_p),
        ("v_cache_p", v_cache_p),
        ("q_d", q_d),
        ("k_cache_d", k_cache_d),
        ("v_cache_d", v_cache_d),
    ]:
        if x is None:
            raise ValueError(f"{name} must not be None for TileLang fused backend.")
        if not x.is_cuda:
            raise ValueError(f"{name} must be CUDA tensor.")
        if x.dtype != torch.float16:
            raise ValueError(f"{name} must be float16.")
        if x.ndim != 4:
            raise ValueError(f"{name} must be rank-4: [B,S,H,D].")
        if x.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous on last dimension.")

    if q_p.shape[-1] != q_d.shape[-1]:
        raise ValueError("prefill/decode head dim mismatch.")
    if q_p.shape[2] != q_d.shape[2]:
        raise ValueError("prefill/decode head count mismatch.")
    if k_cache_p.shape[2] != q_p.shape[2] or k_cache_d.shape[2] != q_d.shape[2]:
        raise ValueError("TileLang fused backend currently requires Q heads == KV heads.")
    if v_cache_p.shape[-1] != q_p.shape[-1] or v_cache_d.shape[-1] != q_d.shape[-1]:
        raise ValueError("TileLang fused backend currently requires d_v == d_q.")

@lru_cache(maxsize=64)
def _build_fused_kernel(
    batch_p: int,
    seq_q_p: int,
    seq_kv_p: int,
    batch_d: int,
    seq_q_d: int,
    seq_kv_d: int,
    heads: int,
    dim: int,
    num_splits_p: int,
    num_splits_d: int,
    block_m_p: int,
    block_n_p: int,
    block_m_d: int,
    block_n_d: int,
    threads: int,
    fused_op: int,
    causal_prefill: bool,
    softmax_scale: float,
):
    _require_tilelang()
    sm_num = driver.get_num_sms()
    sm_scale = float(softmax_scale * 1.44269504)

    q_tiles_p = _ceildiv(seq_q_p, block_m_p)
    q_tiles_d = _ceildiv(seq_q_d, block_m_d)
    prefill_blocks = q_tiles_p * batch_p * heads * num_splits_p
    decode_blocks = q_tiles_d * batch_d * heads * num_splits_d
    prefill_slots = prefill_blocks
    decode_slots = decode_blocks
    total_rounds = prefill_slots + decode_slots + 1
    past_len_prefill = max(0, seq_kv_p - seq_q_p)

    total_n_blocks_p = _ceildiv(seq_kv_p, block_n_p)
    total_n_blocks_d = _ceildiv(seq_kv_d, block_n_d)
    n_blocks_per_split_p = _ceildiv(total_n_blocks_p, num_splits_p)
    n_blocks_per_split_d = _ceildiv(total_n_blocks_d, num_splits_d)

    @tilelang.jit(
        out_idx=[11, 12],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )
    def _kernel():
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            Qp: T.Tensor([batch_p, seq_q_p, heads, dim], dtype),
            Kp: T.Tensor([batch_p, seq_kv_p, heads, dim], dtype),
            Vp: T.Tensor([batch_p, seq_kv_p, heads, dim], dtype),
            Qd: T.Tensor([batch_d, seq_q_d, heads, dim], dtype),
            Kd: T.Tensor([batch_d, seq_kv_d, heads, dim], dtype),
            Vd: T.Tensor([batch_d, seq_kv_d, heads, dim], dtype),
            SchedState: T.Tensor([sm_num + 2], T.int32),
            LseP: T.Tensor([batch_p, heads, num_splits_p, seq_q_p], accum_dtype),
            PartP: T.Tensor([batch_p, heads, num_splits_p, seq_q_p, dim], accum_dtype),
            LseD: T.Tensor([batch_d, heads, num_splits_d, seq_q_d], accum_dtype),
            PartD: T.Tensor([batch_d, heads, num_splits_d, seq_q_d, dim], accum_dtype),
            OutP: T.Tensor([batch_p, seq_q_p, heads, dim], dtype),
            OutD: T.Tensor([batch_d, seq_q_d, heads, dim], dtype),
        ):
            with T.Kernel(sm_num, threads=threads) as (sm_slot):
                # SchedState layout:
                # [0:sm_num)      -> per-slot op tag counters (simulates per-SM counters)
                # [sm_num + 0]    -> global prefill queue cursor
                # [sm_num + 1]    -> global decode queue cursor
                # TileLang doesn't expose %smid directly in Python DSL, so we map one persistent
                # cooperative block per "slot" via T.Kernel(sm_num, ...).
                tx = T.get_thread_binding()

                valid_shared = T.alloc_shared([1], T.int32)
                op_shared = T.alloc_shared([1], T.int32)
                linear_id_shared = T.alloc_shared([1], T.int32)
                done_shared = T.alloc_shared([1], T.int32)

                Qp_shared = T.alloc_shared([block_m_p, dim], dtype)
                Kp_shared = T.alloc_shared([block_n_p, dim], dtype)
                Vp_shared = T.alloc_shared([block_n_p, dim], dtype)
                acc_sp = T.alloc_fragment([block_m_p, block_n_p], accum_dtype)
                acc_sp_cast = T.alloc_fragment([block_m_p, block_n_p], dtype)
                acc_op = T.alloc_fragment([block_m_p, dim], accum_dtype)
                scores_max_p = T.alloc_fragment([block_m_p], accum_dtype)
                scores_max_prev_p = T.alloc_fragment([block_m_p], accum_dtype)
                scores_scale_p = T.alloc_fragment([block_m_p], accum_dtype)
                scores_sum_p = T.alloc_fragment([block_m_p], accum_dtype)
                logsum_p = T.alloc_fragment([block_m_p], accum_dtype)
                oob_p = T.alloc_var(T.bool)
                has_valid_p = T.alloc_var(T.bool)

                Qd_shared = T.alloc_shared([block_m_d, dim], dtype)
                Kd_shared = T.alloc_shared([block_n_d, dim], dtype)
                Vd_shared = T.alloc_shared([block_n_d, dim], dtype)
                acc_sd = T.alloc_fragment([block_m_d, block_n_d], accum_dtype)
                acc_sd_cast = T.alloc_fragment([block_m_d, block_n_d], dtype)
                acc_od = T.alloc_fragment([block_m_d, dim], accum_dtype)
                scores_max_d = T.alloc_fragment([block_m_d], accum_dtype)
                scores_max_prev_d = T.alloc_fragment([block_m_d], accum_dtype)
                scores_scale_d = T.alloc_fragment([block_m_d], accum_dtype)
                scores_sum_d = T.alloc_fragment([block_m_d], accum_dtype)
                logsum_d = T.alloc_fragment([block_m_d], accum_dtype)
                has_valid_d = T.alloc_var(T.bool)

                for _ in T.serial(total_rounds):
                    if tx == 0:
                        if (SchedState[sm_num + 0] >= prefill_slots) and (SchedState[sm_num + 1] >= decode_slots):
                            done_shared[0] = 1
                        else:
                            done_shared[0] = 0
                        valid_shared[0] = 0
                        op_shared[0] = 0
                        linear_id_shared[0] = 0
                        if done_shared[0] == 0:
                            if prefill_slots == 0 and decode_slots == 0:
                                valid_shared[0] = 0
                            else:
                                if prefill_slots == 0:
                                    op_shared[0] = 1
                                elif decode_slots == 0:
                                    op_shared[0] = 0
                                elif fused_op & 1:
                                    sched_tag = T.atomic_add(SchedState[sm_slot], 1, return_prev=True)
                                    if prefill_slots <= decode_slots:
                                        total_tags = decode_slots // prefill_slots + 1
                                        if sched_tag % total_tags == 0:
                                            op_shared[0] = 0
                                        else:
                                            op_shared[0] = 1
                                    else:
                                        pref_tags = prefill_slots // decode_slots
                                        if sched_tag % (pref_tags + 1) < pref_tags:
                                            op_shared[0] = 0
                                        else:
                                            op_shared[0] = 1
                                else:
                                    sched_tag = T.atomic_add(SchedState[sm_slot], 1, return_prev=True)
                                    op_shared[0] = sched_tag % 2

                                if op_shared[0] == 0:
                                    linear_id_shared[0] = T.atomic_add(SchedState[sm_num + 0], 1, return_prev=True)
                                else:
                                    linear_id_shared[0] = T.atomic_add(SchedState[sm_num + 1], 1, return_prev=True)
                                if op_shared[0] == 0 and linear_id_shared[0] >= prefill_slots:
                                    op_shared[0] = 1
                                    linear_id_shared[0] = T.atomic_add(SchedState[sm_num + 1], 1, return_prev=True)
                                elif op_shared[0] == 1 and linear_id_shared[0] >= decode_slots:
                                    op_shared[0] = 0
                                    linear_id_shared[0] = T.atomic_add(SchedState[sm_num + 0], 1, return_prev=True)

                                if (op_shared[0] == 0 and linear_id_shared[0] < prefill_slots) or (
                                    op_shared[0] == 1 and linear_id_shared[0] < decode_slots
                                ):
                                    valid_shared[0] = 1
                                else:
                                    valid_shared[0] = 0

                    T.sync_threads()
                    if done_shared[0] == 0 and valid_shared[0] == 1 and op_shared[0] == 0:
                        p_linear = linear_id_shared[0]
                        m_block = p_linear % q_tiles_p
                        split_idx = (p_linear // q_tiles_p) % num_splits_p
                        bid_p = (p_linear // q_tiles_p // num_splits_p) % batch_p
                        hid_p = (p_linear // q_tiles_p // num_splits_p // batch_p) % heads
                        m_start = m_block * block_m_p

                        n_block_min = split_idx * n_blocks_per_split_p
                        n_block_max = T.min(total_n_blocks_p, (split_idx + 1) * n_blocks_per_split_p)
                        if causal_prefill:
                            n_block_max = T.min(
                                n_block_max,
                                _ceildiv((m_block + 1) * block_m_p + past_len_prefill, block_n_p),
                            )
                        has_valid_p = n_block_max > n_block_min

                        for i, j in T.Parallel(block_m_p, dim):
                            q_idx = m_start + i
                            Qp_shared[i, j] = T.if_then_else(q_idx < seq_q_p, Qp[bid_p, q_idx, hid_p, j], 0)
                        T.fill(acc_op, 0)
                        T.fill(logsum_p, 0)
                        T.fill(scores_max_p, -T.infinity(accum_dtype))

                        if has_valid_p:
                            for kb in T.serial(total_n_blocks_p):
                                if kb >= n_block_min and kb < n_block_max:
                                    k_start = kb * block_n_p
                                    for i, j in T.Parallel(block_n_p, dim):
                                        k_idx = k_start + i
                                        Kp_shared[i, j] = T.if_then_else(k_idx < seq_kv_p, Kp[bid_p, k_idx, hid_p, j], 0)
                                        Vp_shared[i, j] = T.if_then_else(k_idx < seq_kv_p, Vp[bid_p, k_idx, hid_p, j], 0)

                                    for i, j in T.Parallel(block_m_p, block_n_p):
                                        q_idx = m_start + i + past_len_prefill
                                        k_idx = k_start + j
                                        oob_p = (m_start + i >= seq_q_p) | (k_idx >= seq_kv_p)
                                        if causal_prefill:
                                            acc_sp[i, j] = T.if_then_else(
                                                oob_p,
                                                -T.infinity(accum_dtype),
                                                T.if_then_else(q_idx >= k_idx, 0, -T.infinity(accum_dtype)),
                                            )
                                        else:
                                            acc_sp[i, j] = T.if_then_else(oob_p, -T.infinity(accum_dtype), 0)

                                    T.gemm(Qp_shared, Kp_shared, acc_sp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                    T.copy(scores_max_p, scores_max_prev_p)
                                    T.fill(scores_max_p, -T.infinity(accum_dtype))
                                    T.reduce_max(acc_sp, scores_max_p, dim=1, clear=False)
                                    for i in T.Parallel(block_m_p):
                                        scores_max_p[i] = T.max(scores_max_p[i], scores_max_prev_p[i])
                                        scores_scale_p[i] = T.exp2(scores_max_prev_p[i] * sm_scale - scores_max_p[i] * sm_scale)
                                    for i, j in T.Parallel(block_m_p, block_n_p):
                                        acc_sp[i, j] = T.exp2(acc_sp[i, j] * sm_scale - scores_max_p[i] * sm_scale)
                                    T.reduce_sum(acc_sp, scores_sum_p, dim=1)
                                    for i in T.Parallel(block_m_p):
                                        logsum_p[i] = logsum_p[i] * scores_scale_p[i] + scores_sum_p[i]
                                    for ii in T.serial(block_m_p):
                                        for jj in T.serial(block_n_p):
                                            acc_sp_cast[ii, jj] = T.cast(acc_sp[ii, jj], dtype)
                                    for i, j in T.Parallel(block_m_p, dim):
                                        acc_op[i, j] *= scores_scale_p[i]
                                    T.gemm(acc_sp_cast, Vp_shared, acc_op, policy=T.GemmWarpPolicy.FullRow)

                        for i in T.Parallel(block_m_p):
                            q_idx = m_start + i
                            if q_idx < seq_q_p:
                                if has_valid_p:
                                    LseP[bid_p, hid_p, split_idx, q_idx] = T.log2(logsum_p[i]) + scores_max_p[i] * sm_scale
                                else:
                                    LseP[bid_p, hid_p, split_idx, q_idx] = -T.infinity(accum_dtype)
                        for i, j in T.Parallel(block_m_p, dim):
                            q_idx = m_start + i
                            if q_idx < seq_q_p:
                                PartP[bid_p, hid_p, split_idx, q_idx, j] = T.if_then_else(has_valid_p, acc_op[i, j] / logsum_p[i], 0)

                    elif done_shared[0] == 0 and valid_shared[0] == 1 and op_shared[0] == 1:
                        d_linear = linear_id_shared[0]
                        m_block = d_linear % q_tiles_d
                        split_idx = (d_linear // q_tiles_d) % num_splits_d
                        bid_d = (d_linear // q_tiles_d // num_splits_d) % batch_d
                        hid_d = (d_linear // q_tiles_d // num_splits_d // batch_d) % heads
                        m_start = m_block * block_m_d

                        n_block_min = split_idx * n_blocks_per_split_d
                        n_block_max = T.min(total_n_blocks_d, (split_idx + 1) * n_blocks_per_split_d)
                        has_valid_d = n_block_max > n_block_min

                        for i, j in T.Parallel(block_m_d, dim):
                            q_idx = m_start + i
                            Qd_shared[i, j] = T.if_then_else(q_idx < seq_q_d, Qd[bid_d, q_idx, hid_d, j], 0)
                        T.fill(acc_od, 0)
                        T.fill(logsum_d, 0)
                        T.fill(scores_max_d, -T.infinity(accum_dtype))

                        if has_valid_d:
                            for kb in T.serial(total_n_blocks_d):
                                if kb >= n_block_min and kb < n_block_max:
                                    k_start = kb * block_n_d
                                    for i, j in T.Parallel(block_n_d, dim):
                                        k_idx = k_start + i
                                        Kd_shared[i, j] = T.if_then_else(k_idx < seq_kv_d, Kd[bid_d, k_idx, hid_d, j], 0)
                                        Vd_shared[i, j] = T.if_then_else(k_idx < seq_kv_d, Vd[bid_d, k_idx, hid_d, j], 0)

                                    for i, j in T.Parallel(block_m_d, block_n_d):
                                        q_idx = m_start + i
                                        k_idx = k_start + j
                                        acc_sd[i, j] = T.if_then_else(
                                            (q_idx < seq_q_d) & (k_idx < seq_kv_d),
                                            0,
                                            -T.infinity(accum_dtype),
                                        )

                                    T.gemm(Qd_shared, Kd_shared, acc_sd, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                                    T.copy(scores_max_d, scores_max_prev_d)
                                    T.fill(scores_max_d, -T.infinity(accum_dtype))
                                    T.reduce_max(acc_sd, scores_max_d, dim=1, clear=False)
                                    for i in T.Parallel(block_m_d):
                                        scores_max_d[i] = T.max(scores_max_d[i], scores_max_prev_d[i])
                                        scores_scale_d[i] = T.exp2(scores_max_prev_d[i] * sm_scale - scores_max_d[i] * sm_scale)
                                    for i, j in T.Parallel(block_m_d, block_n_d):
                                        acc_sd[i, j] = T.exp2(acc_sd[i, j] * sm_scale - scores_max_d[i] * sm_scale)
                                    T.reduce_sum(acc_sd, scores_sum_d, dim=1)
                                    for i in T.Parallel(block_m_d):
                                        logsum_d[i] = logsum_d[i] * scores_scale_d[i] + scores_sum_d[i]
                                    for ii in T.serial(block_m_d):
                                        for jj in T.serial(block_n_d):
                                            acc_sd_cast[ii, jj] = T.cast(acc_sd[ii, jj], dtype)
                                    for i, j in T.Parallel(block_m_d, dim):
                                        acc_od[i, j] *= scores_scale_d[i]
                                    T.gemm(acc_sd_cast, Vd_shared, acc_od, policy=T.GemmWarpPolicy.FullRow)

                        for i in T.Parallel(block_m_d):
                            q_idx = m_start + i
                            if q_idx < seq_q_d:
                                if has_valid_d:
                                    LseD[bid_d, hid_d, split_idx, q_idx] = T.log2(logsum_d[i]) + scores_max_d[i] * sm_scale
                                else:
                                    LseD[bid_d, hid_d, split_idx, q_idx] = -T.infinity(accum_dtype)
                        for i, j in T.Parallel(block_m_d, dim):
                            q_idx = m_start + i
                            if q_idx < seq_q_d:
                                PartD[bid_d, hid_d, split_idx, q_idx, j] = T.if_then_else(has_valid_d, acc_od[i, j] / logsum_d[i], 0)

                    T.sync_threads()
            with T.Kernel(_ceildiv(seq_q_p, block_m_p), heads, batch_p, threads=128) as (bx, hid, bid):
                lse_local_split = T.alloc_var(accum_dtype)
                lse_logsum_local = T.alloc_var(accum_dtype)
                lse_max_local = T.alloc_var(accum_dtype)
                scale_local = T.alloc_var(accum_dtype)
                for i in T.serial(block_m_p):
                    q_idx = bx * block_m_p + i
                    if q_idx < seq_q_p:
                        lse_max_local = -T.infinity(accum_dtype)
                        for s in T.serial(num_splits_p):
                            lse_max_local = T.max(lse_max_local, LseP[bid, hid, s, q_idx])
                        if lse_max_local <= -1e20:
                            for j in T.Parallel(dim):
                                OutP[bid, q_idx, hid, j] = 0
                        else:
                            lse_logsum_local = 0
                            for s in T.serial(num_splits_p):
                                lse_local_split = LseP[bid, hid, s, q_idx]
                                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
                            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
                            for j in T.Parallel(dim):
                                OutP[bid, q_idx, hid, j] = 0
                            for s in T.serial(num_splits_p):
                                lse_local_split = LseP[bid, hid, s, q_idx]
                                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                                for j in T.Parallel(dim):
                                    OutP[bid, q_idx, hid, j] += PartP[bid, hid, s, q_idx, j] * scale_local

            with T.Kernel(_ceildiv(seq_q_d, block_m_d), heads, batch_d, threads=128) as (bx, hid, bid):
                lse_local_split = T.alloc_var(accum_dtype)
                lse_logsum_local = T.alloc_var(accum_dtype)
                lse_max_local = T.alloc_var(accum_dtype)
                scale_local = T.alloc_var(accum_dtype)
                for i in T.serial(block_m_d):
                    q_idx = bx * block_m_d + i
                    if q_idx < seq_q_d:
                        lse_max_local = -T.infinity(accum_dtype)
                        for s in T.serial(num_splits_d):
                            lse_max_local = T.max(lse_max_local, LseD[bid, hid, s, q_idx])
                        if lse_max_local <= -1e20:
                            for j in T.Parallel(dim):
                                OutD[bid, q_idx, hid, j] = 0
                        else:
                            lse_logsum_local = 0
                            for s in T.serial(num_splits_d):
                                lse_local_split = LseD[bid, hid, s, q_idx]
                                lse_logsum_local += T.exp2(lse_local_split - lse_max_local)
                            lse_logsum_local = T.log2(lse_logsum_local) + lse_max_local
                            for j in T.Parallel(dim):
                                OutD[bid, q_idx, hid, j] = 0
                            for s in T.serial(num_splits_d):
                                lse_local_split = LseD[bid, hid, s, q_idx]
                                scale_local = T.exp2(lse_local_split - lse_logsum_local)
                                for j in T.Parallel(dim):
                                    OutD[bid, q_idx, hid, j] += PartD[bid, hid, s, q_idx, j] * scale_local

        return main

    return _kernel()

def true_fused_attn_with_kvcache_tilelang(
    q_p: torch.Tensor,
    k_cache_p: torch.Tensor,
    v_cache_p: torch.Tensor,
    q_d: torch.Tensor,
    k_cache_d: torch.Tensor,
    v_cache_d: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens_p: Optional[Union[int, torch.Tensor]] = None,
    cache_seqlens_d: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table_p: Optional[torch.Tensor] = None,
    block_table_d: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits_p: int = 0,
    num_splits_d: int = 0,
    return_softmax_lse: bool = False,
    fused_params: int = 15,
    warmup_compile: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    _require_tilelang()
    _validate_inputs(q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d)

    if k is not None or v is not None:
        raise NotImplementedError("TileLang fused backend does not yet support append KV (k/v).")
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("TileLang fused backend does not yet support rotary.")
    if cache_batch_idx is not None or cache_leftpad is not None:
        raise NotImplementedError("TileLang fused backend does not yet support cache_batch_idx/cache_leftpad.")
    if block_table_p is not None or block_table_d is not None:
        raise NotImplementedError("TileLang fused backend does not yet support paged KV (block_table).")
    if alibi_slopes is not None:
        raise NotImplementedError("TileLang fused backend does not yet support alibi.")
    if softcap != 0.0:
        raise NotImplementedError("TileLang fused backend does not yet support softcap.")
    if window_size != (-1, -1):
        raise NotImplementedError("TileLang fused backend currently supports full attention window only.")
    if rotary_interleaved is not True:
        raise NotImplementedError("rotary_interleaved flag is unused because rotary is not supported.")
    if fused_params not in (8, 9, 10, 11, 15, 64):
        raise ValueError("fused_params must be one of {8, 9, 10, 11, 15, 64}.")

    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q_p, k_cache_p, v_cache_p = [maybe_contiguous(x) for x in (q_p, k_cache_p, v_cache_p)]
    q_d, k_cache_d, v_cache_d = [maybe_contiguous(x) for x in (q_d, k_cache_d, v_cache_d)]

    batch_p, seq_q_p, heads, dim = q_p.shape
    batch_d, seq_q_d, heads_d, dim_d = q_d.shape
    if heads != heads_d or dim != dim_d:
        raise ValueError("prefill and decode shapes mismatch.")

    plan = get_tilelang_fused_launch_plan(
        q_p=q_p,
        k_cache_p=k_cache_p,
        q_d=q_d,
        k_cache_d=k_cache_d,
        fused_params=fused_params,
        num_splits_p=num_splits_p,
        num_splits_d=num_splits_d,
    )
    splits_p = int(plan["num_splits_p"])
    splits_d = int(plan["num_splits_d"])
    chosen_fused_op = int(plan["fused_op_selected"])
    block_m_p = int(plan["block_m_p"])
    block_n_p = int(plan["block_n_p"])
    block_m_d = int(plan["block_m_d"])
    block_n_d = int(plan["block_n_d"])
    threads = int(plan["threads"])

    cache_seqlens_p_t = _parse_cache_seqlens(cache_seqlens_p, batch_p, k_cache_p.shape[1], q_p.device)
    cache_seqlens_d_t = _parse_cache_seqlens(cache_seqlens_d, batch_d, k_cache_d.shape[1], q_d.device)
    if torch.any(cache_seqlens_p_t != k_cache_p.shape[1]) or torch.any(cache_seqlens_d_t != k_cache_d.shape[1]):
        raise NotImplementedError(
            "Variable cache seqlens are not yet supported in this TileLang fused kernel. "
            "Please pass full-length cache or None."
        )

    scale = (dim**-0.5) if softmax_scale is None else float(softmax_scale)
    kernel = _build_fused_kernel(
        batch_p=batch_p,
        seq_q_p=seq_q_p,
        seq_kv_p=k_cache_p.shape[1],
        batch_d=batch_d,
        seq_q_d=seq_q_d,
        seq_kv_d=k_cache_d.shape[1],
        heads=heads,
        dim=dim,
        num_splits_p=splits_p,
        num_splits_d=splits_d,
        block_m_p=block_m_p,
        block_n_p=block_n_p,
        block_m_d=block_m_d,
        block_n_d=block_n_d,
        threads=threads,
        fused_op=chosen_fused_op,
        causal_prefill=bool(causal),
        softmax_scale=scale,
    )
    if warmup_compile:
        _ = kernel.get_kernel_source()

    sm_num = driver.get_num_sms()
    sched_state = torch.zeros((sm_num + 2,), dtype=torch.int32, device=q_p.device)
    lse_p = torch.empty((batch_p, heads, splits_p, seq_q_p), dtype=torch.float32, device=q_p.device)
    part_p = torch.empty((batch_p, heads, splits_p, seq_q_p, dim), dtype=torch.float32, device=q_p.device)
    lse_d = torch.empty((batch_d, heads, splits_d, seq_q_d), dtype=torch.float32, device=q_d.device)
    part_d = torch.empty((batch_d, heads, splits_d, seq_q_d, dim), dtype=torch.float32, device=q_d.device)

    out_p, out_d = kernel(q_p, k_cache_p, v_cache_p, q_d, k_cache_d, v_cache_d, sched_state, lse_p, part_p, lse_d, part_d)
    if return_softmax_lse:
        return out_p, out_d, lse_p, lse_d
    return out_p, out_d


@torch.no_grad()
def smoke_test_tilelang_true_fused(
    batch_p: int = 1,
    seq_q_p: int = 128,
    seq_kv_p: int = 128,
    batch_d: int = 8,
    seq_q_d: int = 1,
    seq_kv_d: int = 1024,
    heads: int = 16,
    dim: int = 128,
    causal: bool = False,
    fused_params: int = 15,
    atol: float = 3e-2,
    rtol: float = 3e-2,
) -> None:
    _require_tilelang()
    device = torch.device("cuda")
    dtype = torch.float16

    q_p = torch.randn(batch_p, seq_q_p, heads, dim, device=device, dtype=dtype)
    k_p = torch.randn(batch_p, seq_kv_p, heads, dim, device=device, dtype=dtype)
    v_p = torch.randn(batch_p, seq_kv_p, heads, dim, device=device, dtype=dtype)

    q_d = torch.randn(batch_d, seq_q_d, heads, dim, device=device, dtype=dtype)
    k_d = torch.randn(batch_d, seq_kv_d, heads, dim, device=device, dtype=dtype)
    v_d = torch.randn(batch_d, seq_kv_d, heads, dim, device=device, dtype=dtype)

    out_p, out_d = true_fused_attn_with_kvcache_tilelang(
        q_p=q_p,
        k_cache_p=k_p,
        v_cache_p=v_p,
        q_d=q_d,
        k_cache_d=k_d,
        v_cache_d=v_d,
        causal=causal,
        fused_params=fused_params,
    )

    q_p_ref = q_p.permute(0, 2, 1, 3)
    k_p_ref = k_p.permute(0, 2, 1, 3)
    v_p_ref = v_p.permute(0, 2, 1, 3)
    ref_p = torch.nn.functional.scaled_dot_product_attention(q_p_ref, k_p_ref, v_p_ref, is_causal=causal)
    ref_p = ref_p.permute(0, 2, 1, 3).contiguous()

    q_d_ref = q_d.permute(0, 2, 1, 3)
    k_d_ref = k_d.permute(0, 2, 1, 3)
    v_d_ref = v_d.permute(0, 2, 1, 3)
    ref_d = torch.nn.functional.scaled_dot_product_attention(q_d_ref, k_d_ref, v_d_ref, is_causal=False)
    ref_d = ref_d.permute(0, 2, 1, 3).contiguous()

    torch.testing.assert_close(out_p, ref_p, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_d, ref_d, atol=atol, rtol=rtol)
