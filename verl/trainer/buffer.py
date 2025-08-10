from ..protocol import DataProto
import random
from copy import deepcopy
import pickle

class Buffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Optional[DataProto] = None
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, new_data: DataProto):
        if self.buffer is None:
            self.buffer = new_data
            self.size = len(new_data)
        else:
            self.buffer = DataProto.concat([self.buffer, new_data])
            self.size += len(new_data)

        # 超出容量则裁剪
        if self.size > self.capacity:
            self._truncate_to_capacity()

    def pop(self, n: int):
        if self.buffer is None or self.size == 0:
            raise ValueError("Buffer is empty.")
        if n > self.size:
            raise ValueError(f"Cannot pop {n} items from buffer of size {self.size}.")
        indices = list(range(n))
        result = self.buffer.select_by_index(indices)
        remaining_indices = list(range(n, self.size))
        self.buffer = self.buffer.select_by_index(remaining_indices) if remaining_indices else None
        self.size -= n

        return result


    def sample(self, n: int) -> DataProto:
        assert self.buffer is not None and self.size > 0, "Buffer is empty"
        if n > self.size:
            raise ValueError(f"Cannot sample {n} items from buffer of size {self.size}")

        sampled_indices = random.sample(range(self.size), n)
        return self.buffer.select_by_index(sampled_indices)

    def _truncate_to_capacity(self):
        """只保留最近的 capacity 条样本（假设时间顺序在 DataProto 中是保留的）"""
        assert self.buffer is not None
        indices = list(range(self.size - self.capacity, self.size))
        self.buffer = self.buffer.select_by_index(indices)
        self.size = self.capacity


    def save_to_file(self, path: str):
        if self.buffer is None:
            raise ValueError("ReplayBuffer is empty; nothing to save.")
        state = {
            "capacity": self.capacity,
            "size": self.size,
            "buffer": self.buffer
        }
        torch.save(state, path)
        print(f"ReplayBuffer saved to {path}")

    # @classmethod
    # def load_from_file(cls, path: str) -> "ReplayBuffer":
    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"ReplayBuffer file not found at {path}")
    #     state = torch.load(path)
    #     rb = cls(capacity=state["capacity"])
    #     rb.size = state["size"]
    #     rb.buffer = state["buffer"]
    #     print(f"ReplayBuffer loaded from {path}, size: {rb.size}")
    #     return rb
