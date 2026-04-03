# POD-Attention Prefill/Decode Fuse Source Guide

## 0. Scope and Goal

This document explains, in source-level detail, how `pod_attn` fuses prefill and decode attention.

Focus:

1. The exact call chain from Python API to CUDA kernel.
2. How prefill/decode tasks are represented and scheduled.
3. How SM-aware scheduling works (`tbAssign`, `%smid`, `%nsmid`).
4. How split-kv, append-kv, and combine phases fit in.
5. What each `fused_params` option means in code.

Main source files:

1. `pod_attn/pod_attn/fused_attn_interface.py`
2. `pod_attn/pod_attn/fused_api.cpp`
3. `pod_attn/pod_attn/fused_api_setup.h`
4. `pod_attn/pod_attn/fused_fwd_launch_template.h`
5. `pod_attn/pod_attn/fused_fwd_kernel.h`
6. `pod_attn/pod_attn/flash.h`
7. `pod_attn/pod_attn/static_switch.h`
8. `pod_attn/setup.py`
9. `pod_attn/pod_attn/truefused_fwd_*.cu` (template instantiations)


## 1. Big Picture: What "fuse" means here

The implementation does **not** build one algebraically merged attention equation for prefill+decode.

Instead, it:

1. Builds two independent parameter packs (`Flash_fwd_params`), one for prefill and one for decode.
2. Launches a single fused kernel that contains both execution paths.
3. Dynamically assigns each CTA to either prefill task or decode task at runtime.
4. Uses per-SM scheduling state to choose task type and maintain load balance.

So this is a **task-level fuse** (row-block task co-scheduling), not formula-level merge.


## 2. API Entry and Dispatch

### 2.1 Python API

`true_fused_attn_with_kvcache` in `fused_attn_interface.py` is the public fused API.

Behavior:

1. If only decode input exists (`q_p is None`), fallback to FlashAttention decode path.
2. If only prefill input exists (`q_d is None`), fallback to FlashAttention prefill path.
3. Only when both exist, call C++ extension entry:
   `fused_attn_cuda.true_fused_fwd_kvcache(...)`.

Reference:

1. `fused_attn_interface.py:12-137`
2. fallback checks at `:40-78`
3. fused call at `:108-136`

Pre-processing done here:

1. force last-dim contiguous tensors
2. convert integer `cache_seqlens_p/d` to int32 tensors
3. pass `num_splits_p`, `num_splits_d`, `fused_params`


### 2.2 C++ PyBind entry

`PYBIND11_MODULE` binds `true_fused_fwd_kvcache` to:

1. `mha_true_fused_fwd_kvcache(...)`

Reference:

1. `fused_api.cpp:490-494`


## 3. Core C++ Frontend (`mha_true_fused_fwd_kvcache`)

Function:

1. `fused_api.cpp:283-487`

### 3.1 Input and config checks

Important constraints:

1. GPU arch must be Ampere+ (`sm8x` or `sm90`) (`:312-316`).
2. `fused_params` allowed: `8, 9, 10, 11, 15, 64` (`:317-319`).
3. Paged KV for decode disallows `cache_batch_idx` (`:329-335`).


### 3.2 Compute initial work estimates and auto split tweaks

Before setup:

1. `num_m_blocks_p = ceil(seqlen_q_p / 128)` (`:354`)
2. `num_m_blocks_d = ceil(seqlen_q_d / 64)` (`:355`)

Split tuning logic:

1. For `fused_params == 9` or specific `15` cases, decode split may be increased to generate at least one wave (`:356-369`).
2. Prefill max split is bounded by `max_splits_p` to reduce memory cost (`:370-372`).
3. If decode already has >= 1 full wave and bit-8 mode is active, prefill split may be forced to 1 (`:373-375`).


### 3.3 Build prefill and decode `Flash_fwd_params`

Calls `setup(...)` twice:

1. prefill setup (`:383-386`)
2. decode setup (`:387-390`)

The setup function handles:

1. dtype/shape checks
2. head-size restriction
3. seqlen group swap optimization for decode-like shapes
4. param packing
5. split-kv accumulation buffers


### 3.4 Decode-only append path

If `k_` and `v_` provided, decode cache append fields are populated:

1. `params_decode.knew_ptr`, `vnew_ptr`
2. `seqlen_knew`
3. all append strides

Reference:

1. `fused_api.cpp:395-427`


### 3.5 Sequence length and batch index semantics

1. `cache_batch_idx` is decode-side cache indexing (`:429-437`).
2. `seqlens_k_p` / `seqlens_k_d` control cumulative vs per-sequence interpretation (`:439-457`).


### 3.6 Final fused call

`params_prefill.fused_params = params_decode.fused_params = fused_params`
then:

1. `run_true_fused_mha_fwd(params_prefill, params_decode, stream)` (`:459-465`)


## 4. Setup Layer Details (`fused_api_setup.h`)

### 4.1 `setup(...)` restrictions and packing

Function:

1. `fused_api_setup.h:242-381`

Key restrictions:

1. dtype must be fp16 (`:264`)
2. head size must be exactly 128 (`:288`)

Other behavior:

1. decode-like query reshaping optimization (`seqlenq_ngroups_swapped`) (`:297-303`)
2. alloc output and lse (`:328-343`)
3. pack all base fields via `set_params_fused_fprop` (`:344-361`)
4. configure split-kv via `set_params_fused_splitkv` (`:366-369`)
5. fill paged-KV metadata (`:370-377`)


### 4.2 `set_params_fused_fprop(...)`

Function:

1. `fused_api_setup.h:15-148`

It writes the full `Flash_fwd_params` base runtime state:

1. pointers and strides for q/k/v/o
2. dimensions (`b,h,h_k,seqlen_q,seqlen_k,d,...`)
3. softmax scales (`scale_softmax`, `scale_softmax_log2`)
4. dropout fields (kept but dropout effectively disabled in POD path)
5. local/causal window fields


### 4.3 Split heuristic

Function:

1. `num_splits_heuristic(...)` at `:156-190`
2. used by `set_params_fused_splitkv(...)` (`:192-220`)

Mechanism:

1. evaluate occupancy efficiency over split candidates
2. pick smallest split reaching >= 85% of best efficiency
3. allocate `softmax_lse_accum` and `oaccum` when split > 1


## 5. Type and Function Contracts (`flash.h`, `static_switch.h`)

### 5.1 `Flash_fwd_params`

Declared in:

1. `flash.h:51-147`

Critical fields for fuse:

1. base q/k/v/o pointers and strides
2. split accum pointers: `oaccum_ptr`, `softmax_lseaccum_ptr`
3. append pointers: `knew_ptr`, `vnew_ptr`
4. decode cache indirection: `cache_batch_idx`
5. paged KV metadata: `block_table`, `page_block_size`
6. split count: `num_splits`
7. fuse config bits: `fused_params`


### 5.2 Static switches

In `static_switch.h`:

1. `HEADDIM_SWITCH` currently only instantiates head-dim 128 path.
2. `FUSED_SWITCH` supports `9,8,10,11,64`.
3. `15` is not in `FUSED_SWITCH` because it is handled earlier as runtime auto-select.


## 6. Launch Layer (`fused_fwd_launch_template.h`)

## 6.1 Wrapper kernels

Three wrapper kernels:

1. `true_fused_tb_fwd_kernel` -> calls `compute_fused_tb_attn`
2. `true_fused_pt_tb_fwd_kernel` -> calls `compute_fused_pt_tb_attn`
3. split/combine support kernels

Reference:

1. `fused_fwd_launch_template.h:51-91`


### 6.2 `run_true_fused_fwd(...)`

Function:

1. `fused_fwd_launch_template.h:350-515`

Main steps:

1. Assert prefill/decode compatibility:
   same `h`, `d`, `seqlen_k` (`:358-361`).
2. Compute total blocks for prefill/decode (`:366-375`).
3. Resolve many compile-time flags:
   `IsEvenMN`, `DecodeAppend_KV`, `UseSplitPrefill`, `UseSplitDecode` (`:389-398`).
4. Allocate scheduler state:
   `tbAssign` of size `(numSMs + 2)` (`:407-410`).
5. Choose kernel style:
   - if `fused_op & 128`: persistent kernel (`:418-433`)
   - else: one-shot dynamic scheduler kernel (`:434-447`)
6. If split used, run combine kernels after main fused kernel:
   prefill combine (`:459-484`) and decode combine (`:487-513`).

Important note:

1. Current public `fused_params` set does not include bit-128 options.
2. So in normal runs it uses `compute_fused_tb_attn`, not persistent path.


### 6.3 `run_true_fused_mha_fwd_hdim128(...)`

Function:

1. `fused_fwd_launch_template.h:517-616`

This picks tile traits and branch family based on `fusedOp` and split status.

Key branches:

1. `fusedOp == 64` -> HFuse path (`:541-556`).
2. `fusedOp & 8` -> POD path with split-aware decode tile selection (`:558-583`).
3. else path -> alternate tile families (`:584-597`).

Within POD path:

1. decode tile:
   - split decode: `(QDim_d, KVDim_d, Warps_d) = (64,128,4)`
   - non-split decode: `(16,32,1)` (`:562-565`)
2. prefill tile:
   - bit-2 set (`fusedOp & 2`): `(64,32,2)` (`:568-571`)
   - bit-4 set (`fusedOp & 4`): `(16,32,1)` (`:571-573`)
   - default: `(128,64,4)` (`:575-576`)


## 7. CUDA Kernel Layer (`fused_fwd_kernel.h`)

## 7.1 Core compute primitives

### A) Non-split row-block

`compute_attn_1rowblock(...)` at `:70-509`

Behavior:

1. Load one `(m_block, bidb, bidh)` tile.
2. Iterate N dimension in reverse.
3. Compute QK, mask, softmax rescale, multiply V.
4. Write output tile and LSE.


### B) Split/append/paged row-block

`compute_attn_1rowblock_splitkv(...)` at `:515-1084`

Adds:

1. split partition bounds (`n_split_idx`, `num_n_splits`) (`:539-547`)
2. early write to accum buffers when no valid block (`:548-586`)
3. optional decode append-kv path (`Append_KV`) with in-place cache write and optional rotary (`:690-794`)
4. paged-KV block-table walking (`:595-605`, `:875-885`, `:914-920`, etc.)
5. output to accum buffers when `Split=true`, direct output otherwise (`:1040-1083`)


### C) Split combine

`combine_attn_seqk_parallel(...)` at `:1595-1776`

It:

1. reads all split LSE/O accum
2. performs logsumexp merge and weighted sum
3. writes final O and final LSE


## 7.2 Fused schedulers

### A) Persistent scheduler

`compute_fused_pt_tb_attn(...)` at `:1222-1406`

This loops forever and repeatedly pulls tasks from shared scheduler state until out-of-range return.

### B) One-shot scheduler (actual default in this build)

`compute_fused_tb_attn(...)` at `:1412-1590`

This is the main path for public options `8/9/10/11/64`.


## 8. SM-aware scheduling: exact algorithm

Function:

1. `compute_fused_tb_attn` (`fused_fwd_kernel.h:1412-1590`)

### 8.1 Build task-space sizes

1. `num_mblocks_p = ceil(seqlen_q_p / kBlockM_p)` (`:1427`)
2. `num_mblocks_d = ceil(seqlen_q_d / kBlockM_d)` (`:1428`)
3. `prefill_blocks` and `decode_blocks` include split factors if active (`:1438-1441`)
4. `blk_factor_p/d` remap one physical CTA into multiple logical tasks when thread counts differ (`:1431-1436`)

### 8.2 Determine current SM and slots

In thread 0:

1. read `num_SMs` from `%nsmid` (`:1450`)
2. read current SM id from `%smid` (`:1451`)
3. compute slot counts:
   `prefill_slots = ceil(prefill_blocks / blk_factor_p)`,
   `decode_slots = ceil(decode_blocks / blk_factor_d)` (`:1452-1453`)

### 8.3 Choose operation type (`op`)

If `FusedOp & 1`:

1. choose proportional scheduling based on `prefill_slots:decode_slots` (`:1455-1477`)

Else:

1. choose equal alternation (`atomicAdd % 2`) (`:1478-1479`)

### 8.4 Allocate actual task id

1. `linear_block_id = atomicAdd(&tbAssign[num_SMs + op], 1)` (`:1483`)
2. if selected op exhausted, switch to opposite queue (`:1485-1491`)
3. broadcast `(linear_block_id, op)` via shared memory (`:1492-1502`)

### 8.5 Execute selected task

If `op == 0` (prefill):

1. optional logical remap via `blk_factor_p` (`:1507-1512`)
2. map linear id -> `(m_block, n_split_idx?, bidb, bidh)` (`:1522-1525`)
3. call split or non-split prefill primitive (`:1527-1539`)

If `op == 1` (decode):

1. optional logical remap via `blk_factor_d` (`:1550-1555`)
2. map linear id similarly (`:1562-1565` or `:1573-1577`)
3. call split/non-split decode primitive (`:1567-1581`)


## 9. `fused_params` semantics in code

Accepted at API:

1. `8, 9, 10, 11, 15, 64`

Runtime meaning:

1. `15`: auto choose 9 or 11 in `run_true_fused_mha_fwd` (`fused_api.cpp:24-53`)
2. `64`: HFuse branch (`fused_fwd_launch_template.h:541-556`)
3. `8/9/10/11`: POD scheduling/tile variants

Bit-level interpretation used by scheduler path:

1. bit-1 (`&1`): proportional (`1`) vs equal (`0`) operation scheduling (`fused_fwd_kernel.h:1455-1479`)
2. bit-2 (`&2`): choose smaller prefill tile family in hdim128 launcher (`fused_fwd_launch_template.h:568-571`)
3. bit-4 (`&4`): choose leanest prefill tile family (`:571-573`)
4. bit-8 (`&8`): select split-aware POD launch branch (`:558`)

Observed combinations:

1. `8`  = bit-8 only
2. `9`  = bit-8 + proportional bit
3. `10` = bit-8 + bit-2
4. `11` = bit-8 + bit-2 + proportional bit

`15` is not passed through `FUSED_SWITCH`; it is resolved earlier to 9 or 11.


## 10. Build/Instantiation mapping

Template instantiation source generation:

1. `generate_kernels.py` defines `FUSED_OPS = [0,8,9,10,11,64]`
2. generate specialization files `truefused_fwd_hdim128_fp16_*`

Build list in this repo:

1. `setup.py:129-158` compiles `fo8/fo9/fo10/fo11/fo64` (+ causal/split variants).
2. Plain `fo0` files are not included in this extension build.


## 11. Critical implementation constraints and assumptions

1. POD fused setup enforces fp16 + head_dim=128 (`fused_api_setup.h:264`, `:288`).
2. Launch layer asserts no ALiBi and no softcap in this path (`fused_fwd_launch_template.h:382-385`).
3. Prefill/decode must share `h`, `d`, and `seqlen_k` (`:359-361`).
4. SM ID based logic assumes `%nsmid` correctness; source warns tested on A100 (`fused_fwd_kernel.h:1448-1450`, `1259-1261`).


## 12. What overlap means concretely

At runtime:

1. Some CTAs execute prefill row-blocks.
2. Some CTAs execute decode row-blocks.
3. Operation choice is made per CTA based on per-SM counters and global queues.

Therefore overlap occurs:

1. across different SMs naturally
2. and also in time on the same SM as successive CTA assignments switch between op types

No single CTA simultaneously computes both prefill and decode in one pass; each CTA executes exactly one branch per assignment.


## 13. Minimal trace map for reading in debugger/profiler

If you need an execution trace path:

1. Python:
   `true_fused_attn_with_kvcache` (`fused_attn_interface.py`)
2. C++:
   `mha_true_fused_fwd_kvcache` (`fused_api.cpp`)
3. C++ launch chooser:
   `run_true_fused_mha_fwd` -> `run_true_fused_mha_fwd_hdim128` (`fused_api.cpp`, `fused_fwd_launch_template.h`)
4. Launch:
   `run_true_fused_fwd` (`fused_fwd_launch_template.h`)
5. CUDA scheduler:
   `compute_fused_tb_attn` (`fused_fwd_kernel.h`)
6. Primitive compute:
   `compute_attn_1rowblock` / `compute_attn_1rowblock_splitkv` (`fused_fwd_kernel.h`)
7. Split finalize:
   `combine_attn_seqk_parallel` (`fused_fwd_kernel.h`)


## 14. Practical mental model

You can model this implementation as:

1. Build two task pools (`prefill_slots`, `decode_slots`).
2. Build one shared scheduler (`tbAssign`).
3. Launch one fused kernel where each CTA:
   - asks scheduler for `(op, task_id)`
   - decodes task_id into row-block coordinates
   - runs the corresponding primitive
4. If split is enabled, run combine kernels.

That is the full prefill/decode fuse mechanism in this codebase.

