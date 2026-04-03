import argparse
import time

import torch

import pod_attn
from pod_attn.tilelang_fused_attn import (
    get_tilelang_fused_launch_plan,
    true_fused_attn_with_kvcache_tilelang,
)


def _bench_cuda(fn, warmup: int, iters: int) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    out = fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, out


def main():
    parser = argparse.ArgumentParser(description="Benchmark POD fused CUDA kernel vs TileLang fused kernel.")
    parser.add_argument("--batch-p", type=int, default=1)
    parser.add_argument("--seq-q-p", type=int, default=1024)
    parser.add_argument("--seq-kv-p", type=int, default=12288)
    parser.add_argument("--batch-d", type=int, default=80)
    parser.add_argument("--seq-q-d", type=int, default=1)
    parser.add_argument("--seq-kv-d", type=int, default=12288)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--fused-params", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--causal-prefill", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--compile-tilelang-first", action="store_true")
    parser.add_argument(
        "--print-scheduler-stats",
        action="store_true",
        help="Print TileLang fused scheduler statistics (prefill/decode cursor and slot counters summary).",
    )
    parser.add_argument(
        "--print-scheduler-slot-counters",
        action="store_true",
        help="Print full per-slot scheduler counters (requires --print-scheduler-stats).",
    )
    parser.add_argument(
        "--tilelang-only",
        action="store_true",
        help="Only run TileLang benchmark (useful when fused CUDA extension is unavailable).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    fused_available = hasattr(pod_attn, "true_fused_attn_with_kvcache")
    if (not fused_available) and (not args.tilelang_only):
        print(
            "warning: pod_attn.true_fused_attn_with_kvcache is unavailable; "
            "switching to --tilelang-only mode."
        )
        args.tilelang_only = True

    device = torch.device("cuda")
    dtype = torch.float16

    q_p = torch.randn(args.batch_p, args.seq_q_p, args.heads, args.dim, device=device, dtype=dtype)
    k_p = torch.randn(args.batch_p, args.seq_kv_p, args.heads, args.dim, device=device, dtype=dtype)
    v_p = torch.randn(args.batch_p, args.seq_kv_p, args.heads, args.dim, device=device, dtype=dtype)
    q_d = torch.randn(args.batch_d, args.seq_q_d, args.heads, args.dim, device=device, dtype=dtype)
    k_d = torch.randn(args.batch_d, args.seq_kv_d, args.heads, args.dim, device=device, dtype=dtype)
    v_d = torch.randn(args.batch_d, args.seq_kv_d, args.heads, args.dim, device=device, dtype=dtype)

    if args.print_plan:
        plan = get_tilelang_fused_launch_plan(
            q_p=q_p,
            k_cache_p=k_p,
            q_d=q_d,
            k_cache_d=k_d,
            fused_params=args.fused_params,
        )
        print("tilelang_launch_plan")
        for k in sorted(plan.keys()):
            print(f"  {k}={plan[k]}")

    # Pre-compile tilelang kernel to exclude JIT compile from timing.
    if args.compile_tilelang_first:
        _ = true_fused_attn_with_kvcache_tilelang(
            q_p=q_p,
            k_cache_p=k_p,
            v_cache_p=v_p,
            q_d=q_d,
            k_cache_d=k_d,
            v_cache_d=v_d,
            causal=args.causal_prefill,
            fused_params=args.fused_params,
            warmup_compile=True,
        )
        torch.cuda.synchronize()

    def run_fused_cuda():
        return pod_attn.true_fused_attn_with_kvcache(
            q_p=q_p,
            k_cache_p=k_p,
            v_cache_p=v_p,
            q_d=q_d,
            k_cache_d=k_d,
            v_cache_d=v_d,
            k=None,
            v=None,
            cache_seqlens_p=None,
            cache_seqlens_d=None,
            cache_batch_idx=None,
            softmax_scale=None,
            causal=args.causal_prefill,
            fused_params=args.fused_params,
        )

    def run_tilelang():
        return true_fused_attn_with_kvcache_tilelang(
            q_p=q_p,
            k_cache_p=k_p,
            v_cache_p=v_p,
            q_d=q_d,
            k_cache_d=k_d,
            v_cache_d=v_d,
            cache_seqlens_p=None,
            cache_seqlens_d=None,
            causal=args.causal_prefill,
            fused_params=args.fused_params,
            warmup_compile=False,
        )

    fused_ms = None
    fused_out_p = None
    fused_out_d = None
    t0 = time.time()
    if not args.tilelang_only:
        fused_ms, (fused_out_p, fused_out_d) = _bench_cuda(run_fused_cuda, warmup=args.warmup, iters=args.iters)
    t1 = time.time()
    tilelang_ms, (tile_out_p, tile_out_d) = _bench_cuda(run_tilelang, warmup=args.warmup, iters=args.iters)
    t2 = time.time()

    if args.tilelang_only:
        print("correctness: SKIPPED (--tilelang-only)")
    elif not args.no_check:
        torch.testing.assert_close(tile_out_p, fused_out_p, atol=args.atol, rtol=args.rtol)
        torch.testing.assert_close(tile_out_d, fused_out_d, atol=args.atol, rtol=args.rtol)
        print(f"correctness: PASS (atol={args.atol}, rtol={args.rtol})")
    else:
        print("correctness: SKIPPED (--no-check)")

    sec = tilelang_ms / 1000.0
    prefill_tokens = args.batch_p * args.seq_q_p
    decode_tokens = args.batch_d * args.seq_q_d
    total_tokens = prefill_tokens + decode_tokens
    prefill_toks_s = prefill_tokens / sec
    decode_toks_s = decode_tokens / sec
    total_toks_s = total_tokens / sec

    print("config")
    print(
        f"  batch_p={args.batch_p} seq_q_p={args.seq_q_p} seq_kv_p={args.seq_kv_p} "
        f"batch_d={args.batch_d} seq_q_d={args.seq_q_d} seq_kv_d={args.seq_kv_d} "
        f"heads={args.heads} dim={args.dim} fused_params={args.fused_params}"
    )
    print("benchmark_result_ms")
    if fused_ms is not None:
        print(f"  fused_cuda={fused_ms:.4f}")
    print(f"  tilelang={tilelang_ms:.4f}")
    if fused_ms is not None:
        print(f"  speedup_tilelang_vs_fused={fused_ms / tilelang_ms:.4f}x")
    print("throughput_tokens_per_s")
    print(f"  prefill={prefill_toks_s:.2f}")
    print(f"  decode={decode_toks_s:.2f}")
    print(f"  total={total_toks_s:.2f}")
    print("timing_wall_s")
    if fused_ms is not None:
        print(f"  fused_cuda_total={t1 - t0:.2f}")
    print(f"  tilelang_total={t2 - t1:.2f}")

    if args.print_scheduler_stats:
        _, _, sched_stats = true_fused_attn_with_kvcache_tilelang(
            q_p=q_p,
            k_cache_p=k_p,
            v_cache_p=v_p,
            q_d=q_d,
            k_cache_d=k_d,
            v_cache_d=v_d,
            cache_seqlens_p=None,
            cache_seqlens_d=None,
            causal=args.causal_prefill,
            fused_params=args.fused_params,
            warmup_compile=False,
            return_scheduler_stats=True,
            print_scheduler_stats=False,
            return_scheduler_slot_counters=args.print_scheduler_slot_counters,
        )
        print("tilelang_scheduler_stats_from_bench")
        for k in sorted(sched_stats.keys()):
            print(f"  {k}={sched_stats[k]}")


if __name__ == "__main__":
    main()
