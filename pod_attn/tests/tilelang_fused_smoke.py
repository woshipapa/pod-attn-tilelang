import argparse

import torch

from pod_attn.tilelang_fused_attn import (
    get_tilelang_fused_launch_plan,
    smoke_test_tilelang_true_fused,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_p", type=int, default=1)
    parser.add_argument("--seq_q_p", type=int, default=128)
    parser.add_argument("--seq_kv_p", type=int, default=128)
    parser.add_argument("--batch_d", type=int, default=8)
    parser.add_argument("--seq_q_d", type=int, default=1)
    parser.add_argument("--seq_kv_d", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--non_causal_prefill", action="store_true")
    parser.add_argument("--print_plan", action="store_true")
    args = parser.parse_args()

    if args.print_plan:
        device = torch.device("cuda")
        q_p = torch.empty((args.batch_p, args.seq_q_p, args.heads, args.dim), device=device, dtype=torch.float16)
        k_p = torch.empty((args.batch_p, args.seq_kv_p, args.heads, args.dim), device=device, dtype=torch.float16)
        q_d = torch.empty((args.batch_d, args.seq_q_d, args.heads, args.dim), device=device, dtype=torch.float16)
        k_d = torch.empty((args.batch_d, args.seq_kv_d, args.heads, args.dim), device=device, dtype=torch.float16)
        plan = get_tilelang_fused_launch_plan(q_p=q_p, k_cache_p=k_p, q_d=q_d, k_cache_d=k_d, fused_params=15)
        print("TileLang fused launch plan:")
        for k in sorted(plan.keys()):
            print(f"  {k}={plan[k]}")

    smoke_test_tilelang_true_fused(
        batch_p=args.batch_p,
        seq_q_p=args.seq_q_p,
        seq_kv_p=args.seq_kv_p,
        batch_d=args.batch_d,
        seq_q_d=args.seq_q_d,
        seq_kv_d=args.seq_kv_d,
        heads=args.heads,
        dim=args.dim,
        causal=not args.non_causal_prefill,
    )
    print("TileLang fused prefill+decode smoke test passed.")


if __name__ == "__main__":
    main()
