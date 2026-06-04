# SimAI DeepSpeed ZeRO Workload 简要说明

本文说明当前在 SimAI 训练 workload 文本格式中对 DeepSpeed ZeRO-1/2/3 的模拟方式、输出条目含义和使用方法。

相关入口：

- SimAI 生成入口：`workload_generator/SimAI_training_workload_generator.py`
- ZeRO SimAI 实现：`workload_generator/SimAI_deepspeed_workload_generator.py`
- 输出位置：`results/workload/*.txt`

## 1. 使用方法

生成 ZeRO-1：

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 1 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero1
```

生成 ZeRO-2：

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 2 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero2
```

生成 contiguous gradients 版本 ZeRO-2：

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 2 --contiguous_gradients --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero2_contig
```

生成 ZeRO-3：

```bash
python -m workload_generator.SimAI_training_workload_generator --frame DeepSpeed --stage 3 --world_size 4 --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero3
```

默认只生成训练迭代主体，不包含 DeepSpeed 初始化通信。若需要把非 AMP 场景下的 init broadcast/allgather 也写入 SimAI 文本，增加：

```bash
--simai_include_non_amp_init
```

该选项只有在未开启 `--amp_enabled` 时生效。

## 2. SimAI 输出格式

DeepSpeed ZeRO SimAI 输出使用现有 `Work_Item` 行格式：

```text
name placeholder forward_compute_time forward_comm forward_comm_size backward_compute_time backward_comm backward_comm_size dp_compute_time dp_comm dp_comm_size process_time
```

当前 ZeRO 通信主要写入 `dp_comm` 和 `dp_comm_size` 字段。`name` 用于标识当前行对应的 ZeRO 阶段、训练阶段和通信/计算语义。

生成器按 `global_batch / (micro_batch * dp_num)` 计算 gradient accumulation 次数。每个 accumulation 内生成 forward/backward 相关条目，最后生成 optimizer step 相关条目。

## 3. 模拟机制

### 3.1 ZeRO-1

ZeRO-1 模拟内容：

- forward 阶段按 mocked model 参数顺序生成 compute 条目
- backward 阶段反向遍历参数
- 梯度按 `reduce_bucket_size` 累积 bucket
- bucket 满时立即插入 `zero1_grad_sync`
- `zero1_grad_sync` 使用 `dp_comm=ALLREDUCE`
- step 阶段生成 overflow 检查和参数 allgather

这里保留了 backward compute 与 gradient allreduce 的交错顺序，因此后续调度器可以看到一定程度的 compute-communication overlap 机会。

### 3.2 ZeRO-2

ZeRO-2 模拟内容：

- forward/backward compute 生成方式与 ZeRO-1 类似
- 非 contiguous gradients：
  - bucket 满时生成 `zero2_grad_sync`
  - 使用 `dp_comm=ALLREDUCE`
- contiguous gradients：
  - 构建 `param_range_map`
  - 将全局 gradient buffer 按 `dp_num` 切分到不同 rank range
  - bucket flush 时按参数覆盖的 rank range 切分通信
  - 因 SimAI `Work_Item` 没有 `dst` 字段，physical 中的 `reduce(dst=rank)` 近似表示为 `dp_comm=REDUCESCATTER`
- step 阶段生成 overflow 检查、grad norm 同步和参数 allgather

### 3.3 ZeRO-3

ZeRO-3 模拟内容：

- 维护每个参数的 allgather 状态：
  - `param.id`
  - `param.ds_persist`
  - `param.has_been_allgather`
- 维护 live parameter 计数和参数访问队列：
  - `current_live_parameters`
  - `_param_queue`
  - `_most_recent_step_id_param_fetched_for`
- forward 阶段：
  - 参数访问前生成 `zero3_forward_param_allgather`
  - 使用 `prefetch_bucket_size` 和 `max_live_parameters` 控制 prefetch
  - 访问后按未来使用情况 release/partition 参数
- backward 阶段：
  - 参数访问前生成 `zero3_backward_param_allgather`
  - 非 norm 参数生成两个 backward compute 条目：input grad 和 weight grad
  - 梯度按 `reduce_bucket_size` 累 bucket
  - bucket flush 时生成 `zero3_grad_reduce_scatter`
- step 阶段：
  - flush 剩余 reduce-scatter bucket
  - 生成 overflow 检查和 grad norm 同步
  - reset live state
  - 对 persistent params 重新生成 allgather

### 3.4 Init 通信

init 通信默认不写入，因为 SimAI 训练 workload 通常更关注 steady-state iteration。

如果传入 `--simai_include_non_amp_init` 且未传入 `--amp_enabled`：

- ZeRO-1/2/3 会生成 `zero{stage}_init_broadcast_model`
- ZeRO-3 会额外生成 `zero3_init_param_allgather`

barrier 不写入 SimAI 文本，因为 `Work_Item` 没有明确的 barrier phase 字段。

## 4. 输出条目含义

命名格式：

```text
zero{stage}_{phase_or_operation}
```

`zero1`、`zero2`、`zero3` 表示 ZeRO stage。`{id}` 是 mocked model 参数在 `model.parameters()` 遍历顺序中的编号，不是真实参数名或 layer id。

### 4.1 通用条目

| 条目 | 含义 | 通信字段 |
| --- | --- | --- |
| `zero{stage}_init_broadcast_model` | 非 AMP init 阶段广播模型参数 | `dp_comm=BROADCAST` |
| `zero{stage}_has_overflow` | step 阶段同步 overflow 标志 | `dp_comm=ALLREDUCE` |

### 4.2 ZeRO-1

| 条目 | 含义 | 通信字段 |
| --- | --- | --- |
| `zero1_forward_param_{id}` | 参数 `{id}` 的 forward compute | 通常为 `NONE` |
| `zero1_backward_param_{id}` | 参数 `{id}` 的 backward input grad compute | 通常为 `NONE` |
| `zero1_backward_param_{id}_weight_grad` | 参数 `{id}` 的 backward weight grad compute | 通常为 `NONE` |
| `zero1_grad_sync` | 梯度 allreduce bucket | `dp_comm=ALLREDUCE` |
| `zero1_param_allgather` | step 阶段参数 allgather | `dp_comm=ALLGATHER` |

### 4.3 ZeRO-2

| 条目 | 含义 | 通信字段 |
| --- | --- | --- |
| `zero2_forward_param_{id}` | 参数 `{id}` 的 forward compute | 通常为 `NONE` |
| `zero2_backward_param_{id}` | 参数 `{id}` 的 backward input grad compute | 通常为 `NONE` |
| `zero2_backward_param_{id}_weight_grad` | 参数 `{id}` 的 backward weight grad compute | 通常为 `NONE` |
| `zero2_grad_sync` | 梯度同步 bucket 或 contiguous range chunk | 非 contiguous 为 `ALLREDUCE`，contiguous 近似为 `REDUCESCATTER` |
| `zero2_grad_norm` | step 阶段同步 gradient norm | `dp_comm=ALLREDUCE` |
| `zero2_param_allgather` | step 阶段参数 allgather | `dp_comm=ALLGATHER` |

### 4.4 ZeRO-3

| 条目 | 含义 | 通信字段 |
| --- | --- | --- |
| `zero3_init_broadcast_model` | 非 AMP init 阶段广播模型参数 | `dp_comm=BROADCAST` |
| `zero3_init_param_allgather` | 非 AMP init 阶段 allgather 参数分片 | `dp_comm=ALLGATHER` |
| `zero3_forward_param_allgather` | forward 参数访问前的 allgather 或 prefetch allgather | `dp_comm=ALLGATHER` |
| `zero3_forward_param_{id}` | 参数 `{id}` 的 forward compute | 通常为 `NONE` |
| `zero3_backward_param_allgather` | backward 参数访问前的 allgather 或 prefetch allgather | `dp_comm=ALLGATHER` |
| `zero3_backward_param_{id}` | 参数 `{id}` 的 backward input grad compute | 通常为 `NONE` |
| `zero3_backward_param_{id}_weight_grad` | 参数 `{id}` 的 backward weight grad compute | 通常为 `NONE` |
| `zero3_grad_reduce_scatter` | 梯度 reduce-scatter bucket | `dp_comm=REDUCESCATTER` |
| `zero3_has_overflow` | step 阶段同步 overflow 标志 | `dp_comm=ALLREDUCE` |
| `zero3_grad_norm` | step 阶段同步 gradient norm | `dp_comm=ALLREDUCE` |
| `zero3_step_persistent_param_allgather` | step 后重新 allgather persistent params | `dp_comm=ALLGATHER` |

## 5. 主要局限

- SimAI `Work_Item` 没有 `src` / `dst` 字段，因此 ZeRO-2 contiguous gradients 中的 `reduce(dst=rank)` 只能近似为 `REDUCESCATTER`。
- physical 路径中的 barrier 没有写入 SimAI 文本。
- 当前 DeepSpeed SimAI 路径使用默认 compute time，不支持 `--aiob_enable` 的 DeepSpeed profiling 数据。
- SimAI 行顺序表达了 compute/communication 的相对顺序，但不是 physical `LogItem` 的逐字段等价转换。
