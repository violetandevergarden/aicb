"""
Shared helpers and compatibility entry point for DeepSpeed SimAI text workload
generation.

The stage-specific implementations live in:
- SimAI_deepspeed_stage1_2_workload_generator.py
- SimAI_deepspeed_stage3_workload_generator.py
"""

import math

from workload_generator.SimAI_work_item import Work_Item


class BaseDeepSpeedSIMAIWorkload:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.workload = []
        self.ga_num = 1
        self.default_compute_time = 1
        self.all_params = list(self.model.parameters())
        self.total_params = sum(param.numel() for param in self.all_params)
        self.param_elem_size = self.all_params[0].elem_size() if self.all_params else 2
        # SimAI text workloads are consumed by simai-flow-scheduler, whose DAG
        # needs compute tasks as anchors for communication dependencies. Keep
        # DeepSpeed SimAI compute enabled by default even though the generic
        # CLI flag defaults to False.
        self.compute_enable = getattr(args, "computation_enable", True)
        if getattr(args, "frame", None) == "DeepSpeed":
            self.compute_enable = True

        self.reduce_bucket_numel = 0
        self.reduce_bucket_bytes = 0
        self.current_live_parameters = 0
        self.param_range_map = self._build_model_gbuf_param_range_map()

    @staticmethod
    def _module_params(module):
        if module is None:
            return []
        return list(module.parameters())

    def _param_bytes(self, params):
        return sum(param.msg_size() for param in params)

    def _append_layer_compute_item(
        self,
        name,
        forward_compute_time=0,
        backward_compute_time=0,
        dp_compute_time=0,
    ):
        if not self.compute_enable:
            return
        self._append_item(
            name=name,
            forward_compute_time=forward_compute_time,
            backward_compute_time=backward_compute_time,
            dp_compute_time=dp_compute_time,
        )

    def _iter_deepspeed_layer_specs(self):
        model = getattr(self.model, "model", None)
        if model is None:
            print("[WARN]: DeepSpeed layer granularity expects model.model; no layer specs generated")
            return

        layernorm_params = []
        embedding_params = []
        transformer_layer_specs = []

        for layer in getattr(model, "layers", []):
            input_layernorm = getattr(layer, "input_layernorm", None)
            if input_layernorm is not None:
                layernorm_params.extend(self._module_params(input_layernorm))

            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None:
                transformer_layer_specs.append({
                    "name": "attention_layer",
                    "params": self._module_params(self_attn),
                })

            post_attention_layernorm = getattr(layer, "post_attention_layernorm", None)
            if post_attention_layernorm is not None:
                layernorm_params.extend(self._module_params(post_attention_layernorm))

            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                transformer_layer_specs.append({
                    "name": "mlp_layer",
                    "params": self._module_params(mlp),
                })

        norm = getattr(model, "norm", None)
        if norm is not None:
            layernorm_params.extend(self._module_params(norm))

        embed_tokens = getattr(model, "embed_tokens", None)
        if embed_tokens is not None:
            embedding_params.extend(self._module_params(embed_tokens))

        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is not None:
            embedding_params.extend(self._module_params(lm_head))

        if layernorm_params:
            yield {
                "name": "layernorm",
                "params": layernorm_params,
            }

        if embedding_params:
            yield {
                "name": "embedding_layer",
                "params": embedding_params,
            }

        yield from transformer_layer_specs

    def _param_name(self, stage, param, index=None):
        param_id = getattr(param, "id", index)
        return f"zero{self.args.stage}_{stage}_param_{param_id}"

    def _append_item(
        self,
        name,
        forward_compute_time=None,
        forward_comm="NONE",
        forward_comm_size=0,
        backward_compute_time=None,
        backward_comm="NONE",
        backward_comm_size=0,
        dp_compute_time=None,
        dp_comm="NONE",
        dp_comm_size=0,
    ):
        default = self.default_compute_time
        self.workload.append(
            Work_Item(
                name=name,
                forward_compute_time=(
                    default if forward_compute_time is None else forward_compute_time
                ),
                forward_comm=forward_comm,
                forward_comm_size=int(forward_comm_size),
                backward_compute_time=(
                    default if backward_compute_time is None else backward_compute_time
                ),
                backward_comm=backward_comm,
                backward_comm_size=int(backward_comm_size),
                dp_compute_time=default if dp_compute_time is None else dp_compute_time,
                dp_comm=dp_comm,
                dp_comm_size=int(dp_comm_size),
            )
        )

    def _append_dp_comm_item(self, name, dp_comm, dp_comm_size):
        self._append_item(
            name=name,
            forward_compute_time=0,
            backward_compute_time=0,
            dp_compute_time=0,
            dp_comm=dp_comm,
            dp_comm_size=dp_comm_size,
        )

    def _append_compute_for_param(self, stage, param, index=None):
        if not self.compute_enable:
            return
        if param.get_shape()[-1] == 1:
            return
        compute_time = self.default_compute_time
        if stage == "forward":
            self._append_item(
                name=self._param_name(stage, param, index),
                forward_compute_time=compute_time,
                backward_compute_time=0,
                dp_compute_time=0,
            )
        elif stage == "backward":
            self._append_item(
                name=self._param_name(stage, param, index),
                forward_compute_time=0,
                backward_compute_time=compute_time,
                dp_compute_time=0,
            )
            self._append_item(
                name=f"{self._param_name(stage, param, index)}_weight_grad",
                forward_compute_time=0,
                backward_compute_time=compute_time,
                dp_compute_time=0,
            )

    def _append_init(self):
        for param in self.all_params:
            self._append_dp_comm_item(
                name=f"zero{self.args.stage}_init_broadcast_model",
                dp_comm="BROADCAST",
                dp_comm_size=param.msg_size(),
            )
        if self.args.stage == 3:
            for param in self.all_params:
                self._append_dp_comm_item(
                    name="zero3_init_param_allgather",
                    dp_comm="ALLGATHER",
                    dp_comm_size=param.msg_size(),
                )

    def _build_model_gbuf_param_range_map(self):
        gbuf_size = sum(param.numel() for param in self.all_params)
        if gbuf_size == 0:
            return {}

        gbuf_partition_size = int(math.ceil(gbuf_size / self.args.dp_num))
        gbuf_world_all_ranges = []
        for rank in range(self.args.dp_num):
            gbuf_world_start = rank * gbuf_partition_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + gbuf_partition_size)
            gbuf_world_all_ranges.append((gbuf_world_start, gbuf_world_end))

        start_idx, rank = 0, 0
        gbuf_world_start, gbuf_world_end = gbuf_world_all_ranges[rank]
        param_range_map = {}
        for param in self.all_params:
            param_range_map[id(param)] = []
            end_idx = start_idx + param.numel()
            param_start_idx = start_idx
            while gbuf_world_end < end_idx:
                param_range_map[id(param)].append(
                    (rank, param_start_idx, gbuf_world_end)
                )
                param_start_idx = gbuf_world_end
                rank += 1
                gbuf_world_start, gbuf_world_end = gbuf_world_all_ranges[rank]
            param_range_map[id(param)].append((rank, param_start_idx, end_idx))
            start_idx = end_idx
        return param_range_map

    def _compute_ga_num(self):
        self.ga_num = self.args.global_batch // (
            self.args.micro_batch * self.args.dp_num
        )
        if self.ga_num < 1:
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )
            self.ga_num = 1

    def _append_optional_init(self):
        include_non_amp_init = getattr(
            self.args, "simai_include_non_amp_init", False
        )
        if include_non_amp_init and not self.args.amp_enabled:
            self._append_init()

    def dump_file(self, filename):
        filename = filename + ".txt"
        pp_comm_value = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        )
        pp_comm = (
            f"pp_comm: {pp_comm_value}"
            if self.args.pipeline_model_parallel != 1
            else "pp_comm: 0"
        )
        with open(filename, "w") as f:
            f.write(
                (
                    f"HYBRID_TRANSFORMER_FWD_IN_BCKWD "
                    f"model_parallel_NPU_group: {self.args.tensor_model_parallel_size} "
                    f"ep: {self.args.expert_model_parallel_size} "
                    f"pp: {self.args.pipeline_model_parallel} "
                    f"vpp: {self.args.pipeline_model_parallel} "
                    f"ga: {self.ga_num} all_gpus: {self.args.world_size} "
                    f"checkpoints: 0 checkpoint_initiates: 0 "
                )
                + pp_comm
                + "\n"
            )

            f.write(str(len(self.workload)) + "\n")
            for item in self.workload:
                f.write(
                    "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])
                    + "\n"
                )


def create_deepspeed_simai_workload(model, args):
    granularity = getattr(args, "simai_deepspeed_granularity", "param")
    if args.stage in (1, 2):
        from workload_generator.SimAI_deepspeed_stage1_2_workload_generator import (
            DeepSpeedSIMAIStage1Or2LayerWorkload,
            DeepSpeedSIMAIStage1Or2Workload,
        )

        if granularity == "layer":
            return DeepSpeedSIMAIStage1Or2LayerWorkload(model, args)
        return DeepSpeedSIMAIStage1Or2Workload(model, args)
    if args.stage == 3:
        from workload_generator.SimAI_deepspeed_stage3_workload_generator import (
            DeepSpeedSIMAIStage3LayerWorkload,
            DeepSpeedSIMAIStage3Workload,
        )

        if granularity == "layer":
            return DeepSpeedSIMAIStage3LayerWorkload(model, args)
        return DeepSpeedSIMAIStage3Workload(model, args)
    raise ValueError(f"Unsupported DeepSpeed ZeRO stage: {args.stage}")


class DeepSpeedSIMAIWorkload:
    def __new__(cls, model, args):
        return create_deepspeed_simai_workload(model, args)


__all__ = [
    "BaseDeepSpeedSIMAIWorkload",
    "DeepSpeedSIMAIWorkload",
    "Work_Item",
    "create_deepspeed_simai_workload",
]
