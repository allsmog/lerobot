"""Fixed inference-only CPU vocabulary storage with explicit device ownership."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class CpuVocabularyPlacement:
    """Keep complete tables and use the original embedding as a CPU gather."""

    def __init__(
        self,
        policy: Any,
        torch: Any,
        device: Any,
        host_reserve_bytes: int,
        *,
        embedding_path: str,
        head_paths: tuple[str, str],
        on_fault: Callable[[], None] | None = None,
    ) -> None:
        if (
            isinstance(host_reserve_bytes, bool)
            or not isinstance(host_reserve_bytes, int)
            or host_reserve_bytes < 0
        ):
            raise ValueError("CPU vocabulary placement needs a nonnegative host reserve")
        if (
            policy.training
            or torch.__future__.get_overwrite_module_params_on_conversion()
            or torch.__future__.get_swap_module_params_on_conversion()
        ):
            raise ValueError("CPU vocabulary placement requires the qualified inference conversion flags")
        paths = (embedding_path, *head_paths)
        if len(paths) != 3 or len(set(paths)) != 3:
            raise ValueError("CPU vocabulary placement needs three distinct modules")
        self.policy, self.torch, self.device = policy, torch, torch.device(device)
        self.embedding_path, self.head_paths = embedding_path, head_paths
        self.modules = {path: policy.get_submodule(path) for path in paths}
        embedding = self.modules[embedding_path]
        heads = [self.modules[path] for path in head_paths]
        if type(embedding) is not torch.nn.Embedding or any(
            type(head) is not torch.nn.Linear for head in heads
        ):
            raise ValueError("CPU vocabulary placement requires genuine embedding and linear modules")
        if (
            embedding.max_norm is not None
            or embedding.sparse
            or embedding.scale_grad_by_freq
            or any(head.bias is not None or head.out_features != embedding.num_embeddings for head in heads)
        ):
            raise ValueError("CPU vocabulary placement requires pure lookup and bias-free output heads")
        if any(
            module.training or module._forward_pre_hooks or module._forward_hooks
            for module in self.modules.values()
        ):
            raise ValueError("CPU vocabulary placement does not support training or custom table hooks")
        self.cpu_names = tuple(path + ".weight" for path in paths)
        params = dict(policy.named_parameters())
        self.parameter_names = frozenset(params)
        if not set(self.cpu_names) <= params.keys():
            raise ValueError("CPU vocabulary weights must have distinct registered parameter names")
        selected = {id(self.modules[path].weight) for path in paths}
        if len(selected) != 3 or any(id(params[name]) not in selected for name in self.cpu_names):
            raise ValueError("CPU vocabulary weight bindings are ambiguous")
        self.signatures = {
            name: (id(params[name]), tuple(params[name].shape), params[name].dtype) for name in self.cpu_names
        }

        def root(param: Any) -> tuple[str, int | None, int]:
            return param.device.type, param.device.index, int(param.untyped_storage().data_ptr())

        roots = {root(params[name]) for name in self.cpu_names}
        if len(roots) != 3 or any(
            name not in self.cpu_names and root(param) in roots for name, param in params.items()
        ):
            raise ValueError("CPU vocabulary storage must not alias another parameter")
        self.cpu_bytes = 0
        for name in self.cpu_names:
            param = params[name]
            if (
                param.device.type != self.device.type
                or (self.device.index is not None and param.device.index != self.device.index)
                or not param.is_contiguous()
                or param.grad is not None
                or param.is_quantized
                or int(param.untyped_storage().nbytes()) != param.numel() * param.element_size()
            ):
                raise ValueError("CPU vocabulary weights must be dense owned resident tensors")
            self.cpu_bytes += int(param.untyped_storage().nbytes())
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("MemAvailable", "MemTotal"):
                memory[key] = int(value.split()[0]) * 1024
        self.reserve_bytes = max(host_reserve_bytes, 1024**3, (memory["MemTotal"] + 9) // 10)
        if memory["MemAvailable"] - self.cpu_bytes < self.reserve_bytes:
            raise MemoryError("CPU vocabulary storage would violate the host reserve")
        self.on_fault = on_fault
        self.faulted = False
        self.gather_calls = 0
        self.head_calls = 0
        self.pre_hooks = {}
        self.post_hooks = {}
        try:
            with torch.no_grad():
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                for module in self.modules.values():
                    module.to("cpu")
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                    torch.cuda.empty_cache()
            # Bind ownership after the initial device conversion. Subsequent
            # shadow cycles keep these CPU parameters on their fixed device.
            moved = dict(policy.named_parameters())
            if moved.keys() != self.parameter_names or any(
                (tuple(moved[name].shape), moved[name].dtype) != self.signatures[name][1:]
                for name in self.cpu_names
            ):
                raise RuntimeError("CPU vocabulary conversion changed parameter topology, shape or dtype")
            self.signatures = {
                name: (id(moved[name]), tuple(moved[name].shape), moved[name].dtype)
                for name in self.cpu_names
            }
            self.pre_hooks[embedding_path] = embedding.register_forward_pre_hook(self._before_lookup)
            self.post_hooks[embedding_path] = embedding.register_forward_hook(self._after_lookup)
            for path in head_paths:
                self.pre_hooks[path] = self.modules[path].register_forward_pre_hook(self._reject_head)
            self.check()
        except BaseException:
            self._fault()
            raise

    def _fault(self) -> None:
        self.faulted = True
        if self.on_fault is not None:
            self.on_fault()

    def check(self) -> None:
        try:
            if self.faulted or self.policy.training:
                raise RuntimeError("CPU vocabulary owner is faulted or training")
            params = dict(self.policy.named_parameters())
            embedding = self.modules[self.embedding_path]
            if (
                params.keys() != self.parameter_names
                or embedding.max_norm is not None
                or embedding.sparse
                or embedding.scale_grad_by_freq
                or any(self.modules[path].bias is not None for path in self.head_paths)
            ):
                raise RuntimeError("CPU vocabulary module behavior or topology changed")
            roots = {int(params[name].untyped_storage().data_ptr()) for name in self.cpu_names}
            if len(roots) != 3 or any(
                name not in self.cpu_names
                and param.device.type == "cpu"
                and int(param.untyped_storage().data_ptr()) in roots
                for name, param in params.items()
            ):
                raise RuntimeError("CPU vocabulary storage ownership changed")
            for path, module in self.modules.items():
                name = path + ".weight"
                param = params.get(name)
                if (
                    self.policy.get_submodule(path) is not module
                    or param is None
                    or (id(param), tuple(param.shape), param.dtype) != self.signatures[name]
                    or param.device.type != "cpu"
                    or param.grad is not None
                    or not param.is_contiguous()
                    or int(param.untyped_storage().nbytes()) != param.numel() * param.element_size()
                    or module.training
                    or set(module._forward_pre_hooks) != {self.pre_hooks[path].id}
                    or set(module._forward_hooks)
                    != ({self.post_hooks[path].id} if path in self.post_hooks else set())
                ):
                    raise RuntimeError("CPU vocabulary parameter or hook ownership changed")
        except BaseException:
            self._fault()
            raise

    def _before_lookup(self, module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        try:
            self.check()
            if self.torch.is_grad_enabled() or len(args) != 1 or not isinstance(args[0], self.torch.Tensor):
                raise ValueError("CPU vocabulary lookup requires an inference token tensor")
            tokens = args[0]
            if tokens.dtype not in (self.torch.int32, self.torch.int64):
                raise ValueError("CPU vocabulary lookup requires integer token indices")
            return (tokens.to("cpu"),)
        except BaseException:
            self._fault()
            raise

    def _after_lookup(self, module: Any, args: tuple[Any, ...], output: Any) -> Any:
        try:
            self.check()
            if (
                not isinstance(output, self.torch.Tensor)
                or output.device.type != "cpu"
                or output.dtype != module.weight.dtype
                or tuple(output.shape) != (*args[0].shape, module.embedding_dim)
            ):
                raise RuntimeError("CPU vocabulary lookup returned an unexpected tensor")
            result = output.to(self.device)
            self.gather_calls += 1
            return result
        except BaseException:
            self._fault()
            raise

    def _reject_head(self, module: Any, args: tuple[Any, ...]) -> None:
        self.head_calls += 1
        self._fault()
        raise RuntimeError("CPU vocabulary output heads are outside the qualified action path")

    def metadata(self) -> dict[str, Any]:
        roots = {
            int(module.weight.untyped_storage().data_ptr()): int(module.weight.untyped_storage().nbytes())
            for module in self.modules.values()
            if module.weight.device.type == "cpu"
        }
        return {
            "prepared": True,
            "faulted": self.faulted,
            "cpu_parameter_names": list(self.cpu_names),
            "cpu_storage_bytes": sum(roots.values()),
            "cpu_storage_roots": len(roots),
            "table_devices": {path: str(module.weight.device) for path, module in self.modules.items()},
            "output_device": str(self.device),
            "gather_calls": self.gather_calls,
            "guarded_head_calls": self.head_calls,
            "host_reserve_bytes": self.reserve_bytes,
        }


def prepare_pi05_cpu_vocabulary(
    policy: Any, torch: Any, device: Any, host_reserve_bytes: int, on_fault: Callable[[], None]
) -> CpuVocabularyPlacement:
    """Bind only the three exact current pi0.5 vocabulary modules."""
    embedding = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens"
    heads = (
        "model.paligemma_with_expert.paligemma.lm_head",
        "model.paligemma_with_expert.gemma_expert.lm_head",
    )
    shapes = ((257152, 2048), (257152, 2048), (257152, 1024))
    if (
        getattr(policy.config, "type", None) != "pi05"
        or torch.device(device).type != "cuda"
        or any(
            getattr(policy.config, key, False)
            for key in ("compile_model", "use_visual_memory", "use_proprioceptive_memory")
        )
    ):
        raise ValueError("fixed CPU vocabulary placement requires owned CUDA pi05")
    for path, shape in zip((embedding, *heads), shapes, strict=True):
        weight = policy.get_submodule(path).weight
        if tuple(weight.shape) != shape or weight.dtype != torch.bfloat16:
            raise ValueError("fixed CPU vocabulary placement requires the qualified table shapes and dtype")
    return CpuVocabularyPlacement(
        policy,
        torch,
        device,
        host_reserve_bytes,
        embedding_path=embedding,
        head_paths=heads,
        on_fault=on_fault,
    )
