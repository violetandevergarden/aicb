import dataclasses


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
