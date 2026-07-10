# AICB DeepSpeed ZeRO 工作负载生成说明

本文说明 `aicb` 中新增的 DeepSpeed ZeRO SimAI 文本工作负载生成能力。其输出供
`simai-flow-scheduler` 读取，用于构建计算与通信任务 DAG。

## 目标与范围

原有 SimAI training workload 主要描述普通数据并行训练。ZeRO 会改变参数、梯度和
优化器状态的分片及通信时机，因此新增了 ZeRO-1、ZeRO-2、ZeRO-3 的工作负载生成。

本实现同时生成计算条目和通信条目。通信条目不是独立的流量清单：scheduler 会根据
条目顺序和类型，将其连接到 forward、backward 和 optimizer 的计算任务上。

支持两种粒度：

| 粒度 | 参数 | 用途 |
| --- | --- | --- |
| 参数级 | `--simai_deepspeed_granularity param` | 默认；保留参数级 ZeRO 行为，适合研究 bucket、参数持久化及 ZeRO-3 prefetch 的近似影响。 |
| 层级 | `--simai_deepspeed_granularity layer` | 使用接近原始 `SIMAI_workload` 的层名，适合作为模型级通信/计算的可读近似。 |

## 实现位置

- `workload_generator/SimAI_training_workload_generator.py`：命令行入口。
- `workload_generator/SimAI_deepspeed_workload_generator.py`：公共辅助逻辑、工厂和输出序列化。
- `workload_generator/SimAI_deepspeed_stage1_2_workload_generator.py`：ZeRO-1/2 实现。
- `workload_generator/SimAI_deepspeed_stage3_workload_generator.py`：ZeRO-3 实现。
- `workload_generator/SimAI_work_item.py`：共享的 `Work_Item` 定义。

工厂按 `--stage` 分派：stage 1/2 使用 stage1_2 生成器，stage 3 使用 stage3
生成器。

## 生成的训练语义

### ZeRO-1 / ZeRO-2

- 保留模型 forward/backward 计算；参数级模式保留参数相关计算条目。
- 在梯度计算之后生成 `zero1_grad_sync` 或 `zero2_grad_sync`。
- step 阶段生成 overflow 检查、梯度范数、参数 all-gather，以及通用的
  `cross_entropy1..3`、`optimizer1..4` 条目。
- 层级模式的同步行带层名，例如 `zero2_grad_sync_attention_layer`。

### ZeRO-3

参数级模式的典型顺序为：

```text
forward parameter all-gather -> forward parameter compute
backward parameter all-gather -> backward input compute
backward weight-gradient compute -> gradient reduce-scatter
step/post operations
```

主要条目包括：

```text
zero3_forward_param_allgather
zero3_forward_param_{id}
zero3_backward_param_allgather
zero3_backward_param_{id}
zero3_backward_param_{id}_weight_grad
zero3_grad_reduce_scatter
zero3_step_grad_reduce_scatter
zero3_step_persistent_param_allgather
```

层级模式按 module 发射如下条目：

```text
zero3_forward_allgather_{layer}
zero3_forward_{layer}
zero3_backward_allgather_{layer}
zero3_backward_{layer}
zero3_backward_{layer}_weight_grad
zero3_grad_reducescatter_{layer}
```

层级模式统一为与 `SIMAI_workload` 相近的抽象：embedding 和 layernorm 被简化，
layernorm 使用 pre-LN 顺序。它是 **unbucketed module approximation**：每个 module
的 forward/backward 参数 all-gather 和 weight-gradient 后的 reduce-scatter 分别建模，
并不等价于将参数级 prefetch、live-parameter、持久化和 bucket flush 精确压缩到层级。

## 使用方法

在 `aicb` 目录运行：

```powershell
cd "D:\paper\flow scheduling\SimAI\aicb"
```

生成 ZeRO-3 参数级 workload：

```powershell
python -m workload_generator.SimAI_training_workload_generator `
  --frame DeepSpeed --stage 3 --gpu_type A100 --world_size 4 `
  --global_batch 4 --micro_batch 1 --num_layers 2 --hidden_size 16 `
  --ffn_hidden_size 64 --num_attention_heads 4 --model_name test_zero3
```

生成 ZeRO-3 层级 workload：

```powershell
python -m workload_generator.SimAI_training_workload_generator `
  --frame DeepSpeed --stage 3 --simai_deepspeed_granularity layer `
  --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 `
  --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 `
  --num_attention_heads 4 --model_name test_zero3_layer
```

将 `--stage` 改为 `1` 或 `2` 可生成对应 ZeRO 版本。输出位于
`results/workload/*.txt`。

默认是参数级模式；显式指定 `--simai_deepspeed_granularity param` 与默认行为相同。
DeepSpeed 路径会强制生成计算行，即使一般 workload 命令中关闭了计算相关选项，原因是
scheduler 需要计算任务作为通信依赖锚点。

## 梯度累积与初始化

当 `global_batch / (micro_batch * world_size) > 1` 时，生成器会在每个非最后的
accumulation micro-batch 后写入零开销的 `zero{stage}_ga_boundary` 元数据行。
它不代表计算或网络流量；scheduler 用它精确划分 GA 分组。使用新 scheduler 模拟
参数级 ZeRO 的多次梯度累积时，应重新生成 workload，不能复用缺少该标记的旧文件。

默认只建模稳态训练 iteration。加入 `--simai_include_non_amp_init` 后，会额外生成模型
初始化同步，例如 `zero{stage}_init_broadcast_model`。该 broadcast 不是 ZeRO 独有，
也不是每个训练 step 的核心流量；只有研究端到端启动时间时才建议保留。

## 注意事项与已知近似

- ZeRO-2 的 contiguous gradients 在文本格式中近似为 `REDUCESCATTER`。真实实现可使用
  定向 reduce；当前格式未记录 destination rank，通信竞争模式可能存在差异。
- ZeRO-3 参数级 prefetch 行不携带精确的预取参数或 bucket 成员 metadata。scheduler 按
  workload 顺序建立保守依赖，不能还原 DeepSpeed 的全部 overlap 策略。
- 一维参数（常见于 norm/bias）可能只有通信行而没有独立参数计算行；这来自当前参数级
  计算抽象，不应将其误解为通信遗漏。
- 文本中的参数 id 不保证连续。生成器会跳过不参与相应计算/通信建模的参数，id 是原参数枚举
  的标识而非输出行号。

## 与 scheduler 的配套要求

使用这些 ZeRO workload 时，需要使用已包含 ZeRO 路径的 `simai-flow-scheduler`。旧版
builder 会把参数 all-gather 错接到 weight-gradient 之后，不能正确反映 ZeRO-3/FSDP 的
通信时序。具体接入和 DAG 语义见 scheduler 目录下的 `ZeRO_模拟器扩展说明.md`。
