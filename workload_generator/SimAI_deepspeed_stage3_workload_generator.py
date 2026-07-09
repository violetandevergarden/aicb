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

    def _append_stage3(self):
        persistent_params = self._mark_persistent_parameters()
        for _ in range(self.ga_num):
            self._append_stage3_forward()
            self._append_stage3_backward()
        self._append_stage3_step(persistent_params)

    def workload_generate(self):
        self._compute_ga_num()
        self._append_optional_init()
        self._append_stage3()
