"""Inference-only CPU parameter copies for request-safe GPU parking.

The server serializes every call and completes device synchronization before
reusing these tensors. Mutable buffers are transferred normally at each park.
This helper does not choose which model runs or bypass resource admission.
An explicit byte cap can retain complete parameter storage groups while parked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ParameterShadow:
    def __init__(
        self,
        policy: Any,
        torch: Any,
        resident_device: Any,
        host_reserve_bytes: int,
        *,
        park_retain_bytes: int = 0,
    ) -> None:
        if (
            isinstance(host_reserve_bytes, bool)
            or not isinstance(host_reserve_bytes, int)
            or host_reserve_bytes < 0
        ):
            raise ValueError("host reserve must be a nonnegative integer byte count")
        if (
            isinstance(park_retain_bytes, bool)
            or not isinstance(park_retain_bytes, int)
            or park_retain_bytes < 0
        ):
            raise ValueError("park retention must be a nonnegative integer byte count")
        if (
            torch.__future__.get_overwrite_module_params_on_conversion()
            or torch.__future__.get_swap_module_params_on_conversion()
        ):
            raise ValueError("parameter shadows require the qualified module conversion flags")
        self.policy, self.torch = policy, torch
        self.device = torch.device(resident_device)
        self.park_retain_bytes = park_retain_bytes
        self.last_retained_storage_bytes = 0
        self.last_retained_parameters = 0
        params = dict(policy.named_parameters())
        if not params:
            raise ValueError("parameter shadows require model parameters")
        self.signatures = {name: (tuple(param.shape), param.dtype) for name, param in params.items()}
        self._check(params)
        self.parameter_bytes = sum(param.numel() * param.element_size() for param in params.values())
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemAvailable", "MemTotal"):
                memory[key] = int(value.split()[0]) * 1024
        reserve = max(host_reserve_bytes, 1024**3, (memory["MemTotal"] + 9) // 10)
        if memory["MemAvailable"] - self.parameter_bytes < reserve:
            raise MemoryError("CPU parameter copies would violate the host reserve")
        self.reserve_bytes = reserve
        self.shadows = {}
        pinned = self.device.type == "cuda"
        with torch.no_grad():
            for name, param in params.items():
                copy = torch.empty_like(param, device="cpu", pin_memory=pinned)
                copy.copy_(param, non_blocking=pinned)
                self.shadows[name] = copy
        self._synchronize()
        self.versions = {name: (id(param), int(param._version)) for name, param in params.items()}
        self.last_refreshed = 0

    def _check(self, params: dict[str, Any]) -> None:
        if self.policy.training:
            raise ValueError("parameter shadows support inference mode only")
        if params.keys() != self.signatures.keys():
            raise RuntimeError("model parameter topology changed")
        for name, param in params.items():
            if (tuple(param.shape), param.dtype) != self.signatures[name]:
                raise RuntimeError("model parameter signature changed: " + name)
            if param.layout != self.torch.strided or param.is_quantized or param.grad is not None:
                raise ValueError("parameter shadows require dense parameters without gradients")
            if param.device.type != self.device.type or (
                self.device.index is not None and param.device.index != self.device.index
            ):
                raise RuntimeError("model parameter is not on its resident device: " + name)
            int(param._version)  # Refuse inference tensors without a version counter.

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _retained(self, params: dict[str, Any]) -> tuple[set[int], int]:
        """Keep complete storage groups, including distinct parameters sharing a view."""
        if not self.park_retain_bytes:
            return set(), 0
        groups: dict[tuple[str, int | None, int], tuple[int, list[str]]] = {}
        for name, param in params.items():
            storage = param.untyped_storage()
            size = int(storage.nbytes())
            key = (param.device.type, param.device.index, int(storage.data_ptr()))
            if key not in groups:
                groups[key] = (size, [])
            elif groups[key][0] != size:
                raise RuntimeError("aliased parameter storage changed size")
            groups[key][1].append(name)
        kept: set[int] = set()
        used = 0
        for size, names in sorted(groups.values(), key=lambda group: (-group[0], tuple(group[1]))):
            if size > self.park_retain_bytes - used:
                continue
            kept.update(id(params[name]) for name in names)
            used += size
        return kept, used

    def park(self) -> dict[str, int | bool]:
        params = dict(self.policy.named_parameters())
        self._check(params)
        refreshed = 0
        with self.torch.no_grad():
            for name, param in params.items():
                if self.versions[name] != (id(param), int(param._version)):
                    self.shadows[name].copy_(param, non_blocking=self.device.type == "cuda")
                    refreshed += 1
        self._synchronize()
        by_id = {id(param): self.shadows[name] for name, param in params.items()}
        retained, retained_bytes = self._retained(params)

        def move(tensor: Any) -> Any:
            # Module._apply supplies the actual Parameter objects and buffers.
            # Leave mutable buffers on the normal, value-preserving copy path.
            if id(tensor) in retained:
                return tensor
            return by_id[id(tensor)] if id(tensor) in by_id else tensor.to("cpu")

        self.policy._apply(move)
        self.last_refreshed = refreshed
        self.last_retained_storage_bytes = retained_bytes
        self.last_retained_parameters = len(retained)
        return self.metadata()

    def resumed(self) -> None:
        params = dict(self.policy.named_parameters())
        self._check(params)
        self.versions = {name: (id(param), int(param._version)) for name, param in params.items()}

    def metadata(self) -> dict[str, int | bool]:
        result: dict[str, int | bool] = {
            "parameter_count": len(self.shadows),
            "parameter_bytes": self.parameter_bytes,
            "pinned": self.device.type == "cuda",
            "host_reserve_bytes": self.reserve_bytes,
            "last_refreshed_parameters": self.last_refreshed,
        }
        if self.park_retain_bytes:
            result.update(
                park_retain_limit_bytes=self.park_retain_bytes,
                last_park_retained_storage_bytes=self.last_retained_storage_bytes,
                last_park_retained_parameters=self.last_retained_parameters,
            )
        return result
