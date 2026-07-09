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

from workload_generator.SimAI_deepspeed_workload_generator import (
    BaseDeepSpeedSIMAIWorkload,
)


class DeepSpeedSIMAIStage3Workload(BaseDeepSpeedSIMAIWorkload):
    def __init__(self, model, args):
        super().__init__(model, args)
        self._param_queue = deque()
        self._most_recent_step_id_param_fetched_for = defaultdict(lambda: -1)

    def _reset_param_prefetch_queue(self):
        self._param_queue = deque(
            (param, step_id)
            for step_id, param in enumerate(self.all_params + self.all_params[::-1])
        )
        self._most_recent_step_id_param_fetched_for = defaultdict(lambda: -1)

    def _record_prefetched_param_step(self, param, step_id):
        self._most_recent_step_id_param_fetched_for[param.id] = max(
            step_id,
            self._most_recent_step_id_param_fetched_for[param.id],
        )

    def _drop_stale_param_queue_entries(self, step_id):
        while self._param_queue and self._param_queue[0][1] < step_id:
            stale_param, stale_step_id = self._param_queue.popleft()
            if stale_param.has_been_allgather:
                self._record_prefetched_param_step(stale_param, stale_step_id)
            else:
                print(
                    "WARNING: dropping stale non-prefetched param queue entry "
                    f"{(stale_param.__dict__, stale_step_id)} before step {step_id}"
                )

    def _remove_param_queue_entry(self, param, step_id):
        removed_entry = None
        remaining_entries = deque()
        for entry in self._param_queue:
            queued_param, queued_step_id = entry
            if (
                removed_entry is None
                and queued_param is param
                and queued_step_id == step_id
            ):
                removed_entry = entry
                continue
            remaining_entries.append(entry)
        self._param_queue = remaining_entries
        return removed_entry

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
            self._append_dp_comm_item(name=name, dp_comm="ALLGATHER", dp_comm_size=msg_size)

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

        self._drop_stale_param_queue_entries(step_id)
        if not param.has_been_allgather:
            prefetch_bucket.append(param)
            prefetch_bucket_numel += param.numel()
            prefetch_bucket_bytes += param.msg_size()
            current_entry = self._remove_param_queue_entry(param, step_id)
            if current_entry is None:
                print(
                    "WARNING: expected current param queue entry "
                    f"{(param.__dict__, step_id)} but did not find it"
                )
            param.has_been_allgather = True
            self.current_live_parameters += param.numel()

        while (
            self._param_queue
            and prefetch_bucket_numel < self.args.prefetch_bucket_size
            and self.current_live_parameters < self.args.max_live_parameters
        ):
            future_param, future_step_id = self._param_queue.popleft()
            if future_step_id <= step_id:
                if future_param.has_been_allgather:
                    self._record_prefetched_param_step(future_param, future_step_id)
                else:
                    print(
                        "WARNING: skipping stale non-prefetched future queue entry "
                        f"{(future_param.__dict__, future_step_id)} at step {step_id}"
                    )
                continue
            self._record_prefetched_param_step(future_param, future_step_id)
            if future_param.has_been_allgather:
                continue
            prefetch_bucket.append(future_param)
            future_param.has_been_allgather = True
            self.current_live_parameters += future_param.numel()
            prefetch_bucket_numel += future_param.numel()
            prefetch_bucket_bytes += future_param.msg_size()

        self._append_stage3_allgather(name, prefetch_bucket_bytes)
        self._append_compute_for_param(stage, param)

    def _partition_param(self, param, step_id):
        if param.ds_persist:
            return
        if self._most_recent_step_id_param_fetched_for[param.id] > step_id:
            return
        if not param.has_been_allgather:
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
            self._append_dp_comm_item(
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
                    param, step_id, "backward", "zero3_backward_param_allgather"
                )
            self._partition_param(param, step_id)
            self._reduce_param_with_bucket(param)

        self._reset_param_prefetch_queue()

    def _append_stage3_step(self, persistent_params):
        self._flush_reduce_bucket("zero3_grad_reduce_scatter")
        self._append_dp_comm_item(
            "zero3_has_overflow", dp_comm="ALLREDUCE", dp_comm_size=1
        )
        self._append_dp_comm_item(
            "zero3_grad_norm", dp_comm="ALLREDUCE", dp_comm_size=8
        )

        for param in self.all_params:
            param.has_been_allgather = False
        self.current_live_parameters = 0

        for param in persistent_params:
            self._gather_param_directly(
                param, "step", "zero3_step_persistent_param_allgather"
            )
        self._append_simai_post_items()

    def _append_stage3(self):
        persistent_params = self._mark_persistent_parameters()
        self._reset_param_prefetch_queue()
        for _ in range(self.ga_num):
            self._append_stage3_forward()
            self._append_stage3_backward()
        self._append_stage3_step(persistent_params)

    def workload_generate(self):
        self._compute_ga_num()
        self._append_optional_init()
        self._append_stage3()


class DeepSpeedSIMAIStage3LayerWorkload(BaseDeepSpeedSIMAIWorkload):
    def _append_layer_allgather(self, phase, layer_name, param_bytes):
        if param_bytes <= 0:
            return
        self._append_dp_comm_item(
            name=f"zero3_{phase}_allgather_{layer_name}",
            dp_comm="ALLGATHER",
            dp_comm_size=param_bytes,
        )

    def _append_layer_reduce_scatter(self, layer_name, param_bytes):
        if param_bytes <= 0:
            return
        self._append_dp_comm_item(
            name=f"zero3_grad_reducescatter_{layer_name}",
            dp_comm="REDUCESCATTER",
            dp_comm_size=param_bytes,
        )

    def _append_layer_stage3_forward(self, layer_specs):
        for spec in layer_specs:
            param_bytes = self._param_bytes(spec["params"])
            self._append_layer_allgather("forward", spec["name"], param_bytes)
            self._append_layer_compute_item(
                spec["name"],
                forward_compute_time=self.default_compute_time,
                backward_compute_time=0,
                dp_compute_time=0,
            )

    def _append_layer_stage3_backward(self, layer_specs):
        for spec in reversed(layer_specs):
            param_bytes = self._param_bytes(spec["params"])
            self._append_layer_allgather("backward", spec["name"], param_bytes)
            self._append_layer_compute_item(
                spec["name"],
                forward_compute_time=0,
                backward_compute_time=self.default_compute_time,
                dp_compute_time=self.default_compute_time,
            )
            self._append_layer_reduce_scatter(spec["name"], param_bytes)

    def _append_layer_stage3_step(self):
        self._append_dp_comm_item(
            "zero3_has_overflow", dp_comm="ALLREDUCE", dp_comm_size=1
        )
        self._append_dp_comm_item(
            "zero3_grad_norm", dp_comm="ALLREDUCE", dp_comm_size=8
        )
        self._append_simai_post_items()

    def workload_generate(self):
        self._compute_ga_num()
        self._append_optional_init()
        layer_specs = list(self._iter_deepspeed_layer_specs())
        if not layer_specs:
            print(
                "[WARN]: DeepSpeed layer granularity matched no layers; "
                "falling back to step-only ZeRO items"
            )
        for _ in range(self.ga_num):
            self._append_layer_stage3_forward(layer_specs)
            self._append_layer_stage3_backward(layer_specs)
        self._append_layer_stage3_step()
