"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from collections import defaultdict, deque
import dataclasses
import math


@dataclasses.dataclass
class Work_Item:
    name: str = dataclasses.field(default="none")
    placeholder: int = dataclasses.field(default=-1)
    forward_compute_time: int = dataclasses.field(default=0)
    forward_comm: str = dataclasses.field(default="NONE")
    forward_comm_size: int = dataclasses.field(default=0)
    backward_compute_time: int = dataclasses.field(default=0)
    backward_comm: str = dataclasses.field(default="NONE")
    backward_comm_size: int = dataclasses.field(default=0)
    dp_compute_time: int = dataclasses.field(default=0)
    dp_comm: str = dataclasses.field(default="NONE")
    dp_comm_size: int = dataclasses.field(default=0)
    process_time: int = dataclasses.field(default=100)


class DeepSpeedSIMAIWorkload:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.workload = []
        self.ga_num = 1
        self.default_compute_time = 1
        self.all_params = list(self.model.parameters())
        self.total_params = sum(param.numel() for param in self.all_params)
        self.param_elem_size = self.all_params[0].elem_size() if self.all_params else 2
        self.compute_enable = getattr(args, "computation_enable", True)

        self.reduce_bucket_numel = 0
        self.reduce_bucket_bytes = 0
        self.current_live_parameters = 0
        self._param_queue = deque()
        self._most_recent_step_id_param_fetched_for = defaultdict(lambda: -1)
        self.param_range_map = self._build_model_gbuf_param_range_map()

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
            self._append_item(
                name=f"zero{self.args.stage}_init_broadcast_model",
                dp_comm="BROADCAST",
                dp_comm_size=param.msg_size(),
            )
        if self.args.stage == 3:
            for param in self.all_params:
                self._append_item(
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

    def _append_param_bucketed_dp_comm(self, name, params, bucket_size, comm):
        bucket_numel = 0
        bucket_bytes = 0
        for param in params:
            if bucket_numel and bucket_numel + param.numel() > bucket_size:
                self._append_item(name=name, dp_comm=comm, dp_comm_size=bucket_bytes)
                bucket_numel = 0
                bucket_bytes = 0
            bucket_numel += param.numel()
            bucket_bytes += param.msg_size()
        if bucket_numel:
            self._append_item(name=name, dp_comm=comm, dp_comm_size=bucket_bytes)

    def _append_stage2_contiguous_reduce_bucket(self, bucket):
        rank_start_end_idx = [[-1, -1, -1]]
        for param in bucket[::-1]:
            for rank, start_idx, end_idx in self.param_range_map[id(param)]:
                if rank == rank_start_end_idx[-1][0]:
                    if rank_start_end_idx[-1][-1] != start_idx:
                        print(f"WARNING {rank_start_end_idx[-1]} - {start_idx}")
                    rank_start_end_idx[-1][-1] = end_idx
                else:
                    rank_start_end_idx.append([rank, start_idx, end_idx])

        # SimAI Work_Item cannot express reduce(dst=rank). Keep the physical
        # range chunking, but approximate each directed reduce as DP reduce-scatter.
        for _, start_idx, end_idx in rank_start_end_idx[1:]:
            self._append_item(
                name="zero2_grad_sync",
                dp_comm="REDUCESCATTER",
                dp_comm_size=(end_idx - start_idx) * self.param_elem_size,
            )

    def _append_grad_sync_bucket(self, bucket, bucket_bytes):
        if not bucket:
            return
        if self.args.stage == 1:
            self._append_item(
                name="zero1_grad_sync",
                dp_comm="ALLREDUCE",
                dp_comm_size=bucket_bytes,
            )
        elif self.args.contiguous_gradients:
            self._append_stage2_contiguous_reduce_bucket(bucket)
        else:
            self._append_item(
                name="zero2_grad_sync",
                dp_comm="ALLREDUCE",
                dp_comm_size=bucket_bytes,
            )

    def _append_stage1_or_2_backward(self):
        bucket = []
        bucket_numel = 0
        bucket_bytes = 0
        for index, param in enumerate(self.all_params[::-1]):
            if bucket_numel and param.numel() + bucket_numel > self.args.reduce_bucket_size:
                self._append_grad_sync_bucket(bucket, bucket_bytes)
                bucket = []
                bucket_numel = 0
                bucket_bytes = 0

            bucket.append(param)
            bucket_numel += param.numel()
            bucket_bytes += param.msg_size()
            self._append_compute_for_param("backward", param, index)

        self._append_grad_sync_bucket(bucket, bucket_bytes)

    def _append_total_sharded_allgather(self, name):
        if self.total_params == 0:
            return
        num_shards = max(self.total_params // self.args.allgather_bucket_size, 1)
        shard_size = self.total_params // num_shards
        for index in range(num_shards):
            num_elements = (
                self.total_params - index * shard_size
                if index == num_shards - 1
                else shard_size
            )
            padding_size = (
                self.args.dp_num - num_elements % self.args.dp_num
                if num_elements % self.args.dp_num
                else 0
            )
            num_elements += padding_size
            self._append_item(
                name=name,
                dp_comm="ALLGATHER",
                dp_comm_size=num_elements * self.param_elem_size,
            )

    def _append_stage1_or_2(self):
        for _ in range(self.ga_num):
            for index, param in enumerate(self.all_params):
                self._append_compute_for_param("forward", param, index)
            self._append_stage1_or_2_backward()

        self._append_item(
            f"zero{self.args.stage}_has_overflow",
            dp_comm="ALLREDUCE",
            dp_comm_size=1,
        )
        if self.args.stage == 2:
            self._append_item("zero2_grad_norm", dp_comm="ALLREDUCE", dp_comm_size=8)
        self._append_total_sharded_allgather(f"zero{self.args.stage}_param_allgather")

    def _mark_persistent_parameters(self):
        persistent_params = []
        total_persistent_parameters = 0
        for index, param in enumerate(self.all_params):
            param.id = index
            param.ds_persist = False
            param.has_been_allgather = False
            if (
                param.numel() + total_persistent_parameters
                > self.args.model_persistence_threshold
            ):
                continue
            if param.numel() <= self.args.param_persistence_threshold:
                param.ds_persist = True
                persistent_params.append(param)
                total_persistent_parameters += param.numel()
        return persistent_params

    def _append_stage3_allgather(self, name, msg_size):
        if msg_size:
            self._append_item(name=name, dp_comm="ALLGATHER", dp_comm_size=msg_size)

    def _gather_param_directly(self, param, stage, name):
        if not param.has_been_allgather:
            self._append_stage3_allgather(name, param.msg_size())
            param.has_been_allgather = True
            self.current_live_parameters += param.numel()
        self._append_compute_for_param(stage, param)

    def _gather_param_prefetch(self, param, step_id, stage, name):
        prefetch_bucket = []
        prefetch_bucket_numel = 0
        prefetch_bucket_bytes = 0

        if not param.has_been_allgather:
            while (
                self._param_queue
                and self._param_queue[0][0] != param
                and self._param_queue[0][0].has_been_allgather
            ):
                fetched_param, fetched_step_id = self._param_queue.popleft()
                self._most_recent_step_id_param_fetched_for[fetched_param.id] = max(
                    fetched_step_id,
                    self._most_recent_step_id_param_fetched_for[fetched_param.id],
                )
            prefetch_bucket.append(param)
            prefetch_bucket_numel += param.numel()
            prefetch_bucket_bytes += param.msg_size()
            future_param, future_step_id = self._param_queue.popleft()
            if future_param != param:
                print(
                    f"WARNING: expected {(param.__dict__, step_id)} "
                    f"but got {(future_param.__dict__, future_step_id)}"
                )
            param.has_been_allgather = True
            self.current_live_parameters += param.numel()

        while (
            self._param_queue
            and prefetch_bucket_numel < self.args.prefetch_bucket_size
            and self.current_live_parameters < self.args.max_live_parameters
        ):
            future_param, future_step_id = self._param_queue.popleft()
            self._most_recent_step_id_param_fetched_for[future_param.id] = max(
                future_step_id,
                self._most_recent_step_id_param_fetched_for[future_param.id],
            )
            if future_param.has_been_allgather:
                continue
            prefetch_bucket.append(future_param)
            future_param.has_been_allgather = True
            self.current_live_parameters += future_param.numel()
            prefetch_bucket_numel += future_param.numel()
            prefetch_bucket_bytes += future_param.msg_size()

        self._append_stage3_allgather(name, prefetch_bucket_bytes)
        for bucket_param in prefetch_bucket:
            self._append_compute_for_param(stage, bucket_param)

    def _partition_param(self, param, step_id):
        if len(self._param_queue) == 0:
            param.has_been_allgather = False
            self.current_live_parameters -= param.numel()
            return
        if param.ds_persist:
            return
        if self._most_recent_step_id_param_fetched_for[param.id] > step_id:
            return
        param.has_been_allgather = False
        self.current_live_parameters -= param.numel()

    def _reduce_param_with_bucket(self, param):
        if self.reduce_bucket_numel + param.numel() > self.args.reduce_bucket_size:
            self._flush_reduce_bucket("zero3_grad_reduce_scatter")
        self.reduce_bucket_numel += param.numel()
        self.reduce_bucket_bytes += param.msg_size()

    def _flush_reduce_bucket(self, name):
        if self.reduce_bucket_numel:
            self._append_item(
                name=name,
                dp_comm="REDUCESCATTER",
                dp_comm_size=self.reduce_bucket_bytes,
            )
            self.reduce_bucket_numel = 0
            self.reduce_bucket_bytes = 0

    def _append_stage3_forward(self):
        for index, param in enumerate(self.all_params):
            if len(self._param_queue) == 0:
                self._gather_param_directly(
                    param, "forward", "zero3_forward_param_allgather"
                )
            else:
                self._gather_param_prefetch(
                    param, index, "forward", "zero3_forward_param_allgather"
                )
            self._partition_param(param, index)

    def _append_stage3_backward(self):
        for index, param in enumerate(self.all_params[::-1]):
            step_id = index + len(self.all_params)
            if len(self._param_queue) == 0:
                self._gather_param_directly(
                    param, "backward", "zero3_backward_param_allgather"
                )
            else:
                self._gather_param_prefetch(
                    param, index, "backward", "zero3_backward_param_allgather"
                )
            self._partition_param(param, step_id)
            self._reduce_param_with_bucket(param)

        self._param_queue = deque(
            (param, step_id)
            for step_id, param in enumerate(self.all_params + self.all_params[::-1])
        )
        self._most_recent_step_id_param_fetched_for = defaultdict(lambda: -1)

    def _append_stage3_step(self, persistent_params):
        self._flush_reduce_bucket("zero3_grad_reduce_scatter")
        self._append_item("zero3_has_overflow", dp_comm="ALLREDUCE", dp_comm_size=1)
        self._append_item("zero3_grad_norm", dp_comm="ALLREDUCE", dp_comm_size=8)

        for param in self.all_params:
            param.has_been_allgather = False
        self.current_live_parameters = 0

        for param in persistent_params:
            self._gather_param_directly(
                param, "step", "zero3_step_persistent_param_allgather"
            )

    def _append_stage3(self):
        persistent_params = self._mark_persistent_parameters()
        for _ in range(self.ga_num):
            self._append_stage3_forward()
            self._append_stage3_backward()
        self._append_stage3_step(persistent_params)

    def workload_generate(self):
        self.ga_num = self.args.global_batch // (
            self.args.micro_batch * self.args.dp_num
        )
        if self.ga_num < 1:
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )
            self.ga_num = 1

        include_non_amp_init = getattr(
            self.args, "simai_include_non_amp_init", False
        )
        if include_non_amp_init and not self.args.amp_enabled:
            self._append_init()

        if self.args.stage in (1, 2):
            self._append_stage1_or_2()
        else:
            self._append_stage3()

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
                    f"vpp: {self.args.num_layers} "
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
