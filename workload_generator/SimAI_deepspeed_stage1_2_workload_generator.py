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

from workload_generator.SimAI_deepspeed_workload_generator import (
    BaseDeepSpeedSIMAIWorkload,
)


class DeepSpeedSIMAIStage1Or2Workload(BaseDeepSpeedSIMAIWorkload):
    def _append_param_bucketed_dp_comm(self, name, params, bucket_size, comm):
        bucket_numel = 0
        bucket_bytes = 0
        for param in params:
            if bucket_numel and bucket_numel + param.numel() > bucket_size:
                self._append_dp_comm_item(name=name, dp_comm=comm, dp_comm_size=bucket_bytes)
                bucket_numel = 0
                bucket_bytes = 0
            bucket_numel += param.numel()
            bucket_bytes += param.msg_size()
        if bucket_numel:
            self._append_dp_comm_item(name=name, dp_comm=comm, dp_comm_size=bucket_bytes)

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
            self._append_dp_comm_item(
                name="zero2_grad_sync",
                dp_comm="REDUCESCATTER",
                dp_comm_size=(end_idx - start_idx) * self.param_elem_size,
            )

    def _append_grad_sync_bucket(self, bucket, bucket_bytes):
        if not bucket:
            return
        if self.args.stage == 1:
            self._append_dp_comm_item(
                name="zero1_grad_sync",
                dp_comm="ALLREDUCE",
                dp_comm_size=bucket_bytes,
            )
        elif self.args.contiguous_gradients:
            self._append_stage2_contiguous_reduce_bucket(bucket)
        else:
            self._append_dp_comm_item(
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
            self._append_dp_comm_item(
                name=name,
                dp_comm="ALLGATHER",
                dp_comm_size=num_elements * self.param_elem_size,
            )

    def _append_stage1_or_2(self):
        for _ in range(self.ga_num):
            for index, param in enumerate(self.all_params):
                self._append_compute_for_param("forward", param, index)
            self._append_stage1_or_2_backward()

        self._append_dp_comm_item(
            f"zero{self.args.stage}_has_overflow",
            dp_comm="ALLREDUCE",
            dp_comm_size=1,
        )
        if self.args.stage == 2:
            self._append_dp_comm_item("zero2_grad_norm", dp_comm="ALLREDUCE", dp_comm_size=8)
        self._append_total_sharded_allgather(f"zero{self.args.stage}_param_allgather")

    def workload_generate(self):
        self._compute_ga_num()
        self._append_optional_init()
        self._append_stage1_or_2()
