# SimAI DeepSpeed ZeRO Workload

This document describes the DeepSpeed ZeRO SimAI text workload generator used by
`simai-flow-scheduler`.

## Entry Points

Main command-line entry:

```text
workload_generator/SimAI_training_workload_generator.py
```

DeepSpeed SimAI implementation files:

```text
workload_generator/SimAI_deepspeed_workload_generator.py
workload_generator/SimAI_deepspeed_stage1_2_workload_generator.py
workload_generator/SimAI_deepspeed_stage3_workload_generator.py
```

File responsibilities:

- `simai_work_item.py`: shared SimAI `Work_Item` dataclass.
- `SimAI_deepspeed_workload_generator.py`: shared append/dump/init helpers, the `create_deepspeed_simai_workload` factory, and a compatibility wrapper named `DeepSpeedSIMAIWorkload`.
- `SimAI_deepspeed_stage1_2_workload_generator.py`: ZeRO-1 and ZeRO-2 SimAI text workload logic.
- `SimAI_deepspeed_stage3_workload_generator.py`: ZeRO-3 SimAI text workload logic. This is the main file to extend for FSDP/ZeRO-3 communication timing.

The existing training entry imports the explicit factory:

```python
from workload_generator.SimAI_deepspeed_workload_generator import create_deepspeed_simai_workload
```

`create_deepspeed_simai_workload(model, args)` dispatches by `args.stage`:

```text
stage 1 or 2 -> DeepSpeedSIMAIStage1Or2Workload
stage 3      -> DeepSpeedSIMAIStage3Workload
```

## Usage

Run commands from the `aicb` directory:

```powershell
cd "D:\paper\flow scheduling\SimAI\aicb"
```

Generate ZeRO-1:

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 1 --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero1
```

Generate ZeRO-2:

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 2 --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero2
```

Generate ZeRO-2 with contiguous gradients:

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 2 --contiguous_gradients --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero2_contig
```

Generate ZeRO-3:

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 3 --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero3
```

By default, DeepSpeed SimAI output keeps the existing parameter-level ZeRO
format. This is equivalent to:

```bash
--simai_deepspeed_granularity param
```

To generate layer/module-level output closer to `SIMAI_workload` naming, use:

```bash
--simai_deepspeed_granularity layer
```

Example ZeRO-3 layer-level workload:

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 3 --simai_deepspeed_granularity layer --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero3_layer
```

Example ZeRO-2 layer-level workload:

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 2 --simai_deepspeed_granularity layer --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero2_layer
```

By default, the SimAI text output models steady-state training iteration traffic
and does not include DeepSpeed non-AMP initialization communication. To include
init broadcast/all-gather when AMP is disabled, add:

```bash
--simai_include_non_amp_init
```

## Output Location and Filename

Generated SimAI text workloads are written to:

```text
results/workload/*.txt
```

The filename starts with `--gpu_type`. If `--gpu_type` is not provided, the
generator uses:

```text
unknown_gpu
```

Older generated files may start with `None-...` because `args.gpu_type` defaulted
to Python `None` and was directly formatted into the filename. That prefix only
means no GPU type was provided; it does not affect workload contents.

Example:

```text
A100-test_zero3-world_size4-tp1-pp1-ep1-gbs4-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-False-deepspeed_zero3-param.txt
A100-test_zero3_layer-world_size4-tp1-pp1-ep1-gbs4-mbs1-seq2048-MOE-False-GEMM-False-flash_attn-False-deepspeed_zero3-layer.txt
```

## SimAI Text Format

DeepSpeed ZeRO SimAI output uses the existing 12-field `Work_Item` text format:

```text
name placeholder forward_compute_time forward_comm forward_comm_size backward_compute_time backward_comm backward_comm_size dp_compute_time dp_comm dp_comm_size process_time
```

For DeepSpeed SimAI workloads, compute rows and communication rows are emitted
separately:

```text
zero3_forward_param_allgather      # communication-only row
zero3_forward_param_0              # compute row
zero3_backward_param_allgather     # communication-only row
zero3_backward_param_0             # backward input compute row
zero3_backward_param_0_weight_grad # backward weight compute row
zero3_grad_reduce_scatter          # communication-only row
```

Communication-only rows have all compute fields set to zero:

```text
forward_compute_time = 0
backward_compute_time = 0
dp_compute_time = 0
```

DeepSpeed SimAI compute rows are always emitted for `frame=DeepSpeed`, even if
the generic `--computation_enable` command-line flag is not passed. This is
intentional because `simai-flow-scheduler` needs compute tasks as DAG anchors.

## ZeRO-1

ZeRO-1 behavior:

- Forward compute is emitted in mocked-model parameter order.
- Backward compute is emitted in reverse parameter order.
- Gradient buckets are accumulated by `reduce_bucket_size`.
- Bucket flush emits `zero1_grad_sync` with `dp_comm=ALLREDUCE`.
- Step emits overflow synchronization and parameter all-gather.

Important item names:

```text
zero1_forward_param_{id}
zero1_backward_param_{id}
zero1_backward_param_{id}_weight_grad
zero1_grad_sync
zero1_has_overflow
zero1_param_allgather
```

## ZeRO-2

ZeRO-2 behavior:

- Forward/backward compute generation is similar to ZeRO-1.
- Non-contiguous gradients emit `zero2_grad_sync` with `dp_comm=ALLREDUCE`.
- Contiguous gradients approximate directed reduce operations as `dp_comm=REDUCESCATTER`, because SimAI `Work_Item` has no explicit destination field.
- Step emits overflow synchronization, gradient norm synchronization, and parameter all-gather.

Important item names:

```text
zero2_forward_param_{id}
zero2_backward_param_{id}
zero2_backward_param_{id}_weight_grad
zero2_grad_sync
zero2_has_overflow
zero2_grad_norm
zero2_param_allgather
```

## ZeRO-3

ZeRO-3 parameter-level behavior:

- Parameters are assigned stable ids in `model.parameters()` order.
- Forward emits parameter all-gather before parameter compute.
- Backward emits parameter all-gather before backward parameter compute.
- Backward emits both input-gradient compute and weight-gradient compute for non-1D parameters.
- Gradient bytes are accumulated into reduce-scatter buckets by `reduce_bucket_size`.
- Step flushes remaining reduce-scatter buckets, then emits overflow and grad-norm synchronization.
- Persistent parameters may be all-gathered again after step.

Important item names:

```text
zero3_forward_param_allgather
zero3_forward_param_{id}
zero3_backward_param_allgather
zero3_backward_param_{id}
zero3_backward_param_{id}_weight_grad
zero3_grad_reduce_scatter
zero3_step_grad_reduce_scatter
zero3_has_overflow
zero3_grad_norm
zero3_step_persistent_param_allgather
```

`zero3_step_grad_reduce_scatter` is emitted only when the final parameter-mode
gradient bucket is flushed during the optimizer step. In-loop bucket flushes
retain the name `zero3_grad_reduce_scatter`.

For `ga > 1`, all DeepSpeed SimAI modes emit `zero{stage}_ga_boundary` between
microbatch iterations. It is a metadata-only row with zero compute and
communication, used by the scheduler because prefetch and bucket flushes can
make different GA iterations contain different numbers of rows.

After ZeRO-specific step communication, the SimAI DeepSpeed generator also
appends the generic SimAI post-layer rows:

```text
cross_entropy1
cross_entropy2
cross_entropy3
optimizer1
optimizer2
optimizer3
optimizer4
```

These rows retain the existing `SIMAI_workload` post-step abstraction. Their
order is part of the generated workload contract: ZeRO-specific step traffic
comes first, followed by `cross_entropy*` and `optimizer*`. Consumers must
construct the corresponding dependencies without changing this AICB order.

ZeRO-3 layer-level behavior is an **unbucketed module approximation**. It is
not a compressed form of parameter-level ZeRO-3 and does not model exact
prefetch buckets, live-parameter limits, parameter persistence, or final
bucket flushes.

- Keeps `SIMAI_workload`-style names such as `embedding_layer`, `layernorm`, `attention_layer`, and `mlp_layer`.
- Emits one DP all-gather before each module forward compute.
- Emits one DP all-gather before each module backward compute.
- Emits one DP reduce-scatter after each module backward weight-gradient compute.
- Uses the layer/module parameter byte total as the communication size.
- Does not include parameter ids in item names.

Example item order:

```text
zero3_forward_allgather_layernorm
layernorm
zero3_forward_allgather_embedding_layer
embedding_layer
zero3_forward_allgather_attention_layer
attention_layer
zero3_forward_allgather_mlp_layer
mlp_layer

zero3_backward_allgather_mlp_layer
mlp_layer
zero3_grad_reducescatter_mlp_layer
zero3_backward_allgather_attention_layer
attention_layer
zero3_grad_reducescatter_attention_layer
zero3_backward_allgather_embedding_layer
embedding_layer
zero3_grad_reducescatter_embedding_layer
zero3_backward_allgather_layernorm
layernorm
zero3_grad_reducescatter_layernorm
```

ZeRO-1/2 also support `--simai_deepspeed_granularity layer`, but they have a
different approximation: they keep layer/module compute names and emit
layer/module gradient synchronization rows, rather than ZeRO-3 parameter
all-gathers, such as:

```text
zero1_grad_sync_attention_layer
zero2_grad_sync_attention_layer
zero2_grad_sync_mlp_layer
```

## Why Parameter IDs May Skip Numbers

Parameter ids are assigned from the full mocked model parameter list:

```text
param.id = index in model.parameters()
```

The generator still emits communication for every parameter that ZeRO-3 needs to
all-gather, including 1D norm-like parameters. However, compute rows are skipped
for parameters whose shape has last dimension equal to `1`:

```python
if param.get_shape()[-1] == 1:
    return
```

Therefore output may contain:

```text
zero3_forward_param_0
zero3_forward_param_2
zero3_forward_param_3
...
```

with missing `zero3_forward_param_1`, `zero3_forward_param_6`, etc.

This is expected. Those missing ids usually correspond to layernorm/RMSNorm-like
1D parameters. Their communication can still appear as `zero3_*_param_allgather`
rows, but no separate compute row is generated for them.

For the small test command with `num_layers=2`, the first few parameters are
typically:

```text
param_0  embedding/lm-style matrix parameter -> compute row emitted
param_1  norm-like shape (hidden_size, 1)     -> compute row skipped
param_2  attention q projection              -> compute row emitted
param_3  attention k projection              -> compute row emitted
param_4  attention v projection              -> compute row emitted
param_5  attention o projection              -> compute row emitted
param_6  norm-like shape (hidden_size, 1)     -> compute row skipped
```

The ids are intentionally not renumbered, because keeping original parameter ids
makes it possible to map communication and compute back to the full model
parameter order.

## Current Limitations

- SimAI `Work_Item` has no `src` or `dst` field, so ZeRO-2 contiguous-gradient directed reduce is approximated as reduce-scatter.
- Physical DeepSpeed barriers are not emitted, because `Work_Item` has no explicit barrier phase.
- DeepSpeed SimAI uses default synthetic compute time and does not currently consume `--aiob_enable` profiling output.
- ZeRO-3 prefetch all-gather uses the same item name as ordinary parameter all-gather. The current SimAI text row does not list exact prefetched parameter ids.
- `simai-flow-scheduler` still needs ZeRO/FSDP-aware DAG construction so that `zero3_forward_param_allgather` and `zero3_backward_param_allgather` are treated as compute-before dependencies rather than generic post-weight-gradient `dp_comm`.
