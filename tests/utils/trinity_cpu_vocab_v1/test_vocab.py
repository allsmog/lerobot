"""Model-free CPU fixtures for explicit table ownership, lookup and shadow cycles."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from trinity_noise_padded_v1.test_padded import real_policy_port
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.utils.trinity_http_server import LiberoChunkPolicy
from lerobot.utils.trinity_residency_shadow import ParameterShadow
from lerobot.utils.trinity_vocab_placement import CpuVocabularyPlacement, prepare_pi05_cpu_vocabulary

policy = noise_cases.policy


class SmallTables(torch.nn.Module):
    """Small unit-test tensors, with no checkpoint, CUDA or research model."""

    def __init__(self) -> None:
        super().__init__()
        self.lookup = torch.nn.Embedding(11, 4)
        self.first = torch.nn.Linear(4, 11, bias=False)
        self.second = torch.nn.Linear(3, 11, bias=False)
        self.projection = torch.nn.Linear(4, 3, bias=False)
        self.register_buffer("mutable", torch.tensor([2.0]))
        self.to(dtype=torch.bfloat16)
        self.eval()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(self.lookup(tokens)) + self.mutable


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_text

    def read(path: Path, *args: Any, **kwargs: Any) -> str:
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 100663296 kB\nMemTotal: 134217728 kB\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)


def place(model: SmallTables, **kwargs: Any) -> CpuVocabularyPlacement:
    return CpuVocabularyPlacement(
        model, torch, "cpu", 14 * 1024**3, embedding_path="lookup", head_paths=("first", "second"), **kwargs
    )


def test_exact_lookup_parameters_buffers_and_rng_survive_two_shadow_cycles(memory: None) -> None:
    model = SmallTables()
    tokens = torch.tensor([[0, 10, 2, 2, 7], [9, 1, 0, 3, 8]])
    values = {name: value.detach().clone() for name, value in model.named_parameters()}
    before = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        expected = model(tokens).clone()
    placement = place(model)
    shadow = ParameterShadow(
        model, torch, "cpu", 14 * 1024**3, park_retain_bytes=32, resident_cpu_names=placement.cpu_names
    )
    for _ in range(2):
        with torch.inference_mode():
            assert torch.equal(model(tokens), expected)
        shadow.park()
        placement.check()
        shadow.restore()
        shadow.resumed()
        placement.check()
        assert all(torch.equal(value, values[name]) for name, value in model.named_parameters())
    with torch.inference_mode():
        assert torch.equal(model(tokens), expected)
    meta = shadow.metadata()
    assert meta["resident_cpu_parameter_names"] == sorted(placement.cpu_names)
    assert meta["resident_cpu_storages"] == 3
    assert meta["resident_cpu_storage_bytes"] == sum(
        values[name].numel() * values[name].element_size() for name in placement.cpu_names
    )
    assert meta["last_resident_parameter_storages"] == 1
    assert meta["last_resident_parameter_storage_bytes"] == model.projection.weight.numel() * 2
    assert meta["last_park_retained_parameters"] == 1
    assert model.mutable.item() == 2
    assert placement.gather_calls == 3 and placement.head_calls == 0
    assert torch.equal(before, torch.random.get_rng_state()) and not torch.cuda.is_initialized()


def test_head_invocation_faults_owner_and_prevents_future_lookup(memory: None) -> None:
    model = SmallTables()
    faults = []
    placement = place(model, on_fault=lambda: faults.append(True))
    with pytest.raises(RuntimeError, match="outside the qualified action path"), torch.inference_mode():
        model.first(torch.zeros(1, 4, dtype=torch.bfloat16))
    assert placement.faulted and faults and placement.head_calls == 1
    with pytest.raises(RuntimeError, match="faulted"), torch.inference_mode():
        model.lookup(torch.tensor([[0]]))
    assert placement.gather_calls == 0


@pytest.mark.parametrize("mutation", ["max_norm", "hook", "parameter", "training"])
def test_changed_placement_ownership_or_lookup_behavior_is_refused(memory: None, mutation: str) -> None:
    model = SmallTables()
    placement = place(model)
    if mutation == "max_norm":
        model.lookup.max_norm = 0.1
    elif mutation == "hook":
        model.lookup.register_forward_hook(lambda *_: None)
    elif mutation == "parameter":
        model.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
    elif mutation == "training":
        model.train()
    with pytest.raises(RuntimeError), torch.inference_mode():
        model.lookup(torch.tensor([[0]]))
    assert placement.faulted and placement.gather_calls == 0


@pytest.mark.parametrize("alias", ["head", "outside"])
def test_storage_aliases_are_refused_before_module_moves(memory: None, alias: str) -> None:
    model = SmallTables()
    if alias == "head":
        model.second.weight = torch.nn.Parameter(model.lookup.weight.reshape(-1)[:33].reshape(11, 3))
    else:
        model.projection.weight = torch.nn.Parameter(model.lookup.weight[:3])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="alias"):
        place(model)
    assert not model.lookup._forward_pre_hooks
    assert all(torch.equal(value, before[name]) for name, value in model.named_parameters())


def test_host_reserve_is_checked_before_any_module_conversion(
    memory: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = SmallTables()
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append(True)
        raise AssertionError("module converted before host admission")

    monkeypatch.setattr(torch.nn.Module, "to", forbidden)
    with pytest.raises(MemoryError, match="host reserve"):
        CpuVocabularyPlacement(
            model, torch, "cpu", 2**60, embedding_path="lookup", head_paths=("first", "second")
        )
    assert not calls and not model.lookup._forward_pre_hooks


def test_grad_enabled_lookup_is_refused(memory: None) -> None:
    model = SmallTables()
    placement = place(model)
    with pytest.raises(ValueError, match="inference token tensor"):
        model.lookup(torch.tensor([[0]]))
    assert placement.faulted and placement.gather_calls == 0


@pytest.mark.parametrize(
    "names", [["lookup.weight"], ("missing",), ("lookup.weight", "lookup.weight"), (True,)]
)
def test_shadow_rejects_invalid_declared_cpu_names(memory: None, names: Any) -> None:
    model = SmallTables()
    with pytest.raises(ValueError, match="CPU parameter names"):
        ParameterShadow(model, torch, "cpu", 14 * 1024**3, resident_cpu_names=names)


def test_public_option_and_exact_policy_binding_fail_early() -> None:
    with pytest.raises(ValueError, match="boolean"):
        LiberoChunkPolicy("unused", residency_cpu_vocabulary=1)
    with pytest.raises(ValueError, match="owned residency"):
        LiberoChunkPolicy("unused", residency_cpu_vocabulary=True)
    with pytest.raises(ValueError, match="owned CUDA pi05"):
        prepare_pi05_cpu_vocabulary(
            SimpleNamespace(config=SimpleNamespace(type="pi05")), torch, "cpu", 14 * 1024**3, lambda: None
        )


@pytest.mark.parametrize("key", ["compile_model", "use_visual_memory", "use_proprioceptive_memory"])
def test_cuda_policy_configuration_is_refused_before_any_table_access(key: str) -> None:
    config = SimpleNamespace(type="pi05", **{key: True})
    with pytest.raises(ValueError, match="owned CUDA pi05"):
        prepare_pi05_cpu_vocabulary(SimpleNamespace(config=config), torch, "cuda", 14 * 1024**3, lambda: None)
    assert not torch.cuda.is_initialized()


def cuda_observation_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU-only ports, never GPU allocation, synchronization or hardware evidence."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (100 * 1024**3, 128 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 1024)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: 2048)


@pytest.mark.parametrize("head_fault", [False, True])
def test_actual_server_preparation_and_residency_bridge_preserves_values_or_faults(
    policy: Any, memory: None, monkeypatch: pytest.MonkeyPatch, head_fault: bool
) -> None:
    from lerobot.utils import trinity_vocab_placement as vocab

    server, calls = real_policy_port(policy)
    server.policy.tables = SmallTables()
    server.policy.eval()
    arrays = {**noise_cases.observation(), noise_cases.NOISE_ARRAY: noise_cases.explicit_noise()}
    expected = server.chunk("task", arrays)
    prediction = server.last_policy_chunk.copy()
    server.residency_cpu_vocabulary = True
    server.residency_retain_bytes = 32
    server._residency_shadow = None
    prepared = []

    def cpu_binding(model: Any, tensor_api: Any, device: Any, reserve: int, fault: Any) -> Any:
        assert model is server.policy and device == "cpu" and reserve == 14 * 1024**3
        prepared.append(True)
        return CpuVocabularyPlacement(
            model,
            tensor_api,
            device,
            reserve,
            embedding_path="tables.lookup",
            head_paths=("tables.first", "tables.second"),
            on_fault=fault,
        )

    monkeypatch.setattr(vocab, "prepare_pi05_cpu_vocabulary", cpu_binding)
    cuda_observation_ports(monkeypatch)
    server.prepare_residency_shadow(14 * 1024**3)
    server.prepare_residency_shadow(14 * 1024**3)
    assert prepared == [True]
    original = server.policy.model.sample_actions

    def sample(*args: Any, **kwargs: Any) -> Any:
        if head_fault:
            server.policy.tables.first(torch.zeros(1, 4, dtype=torch.bfloat16))
        else:
            server.policy.tables.lookup(torch.tensor([[0, 10, 2]]))
        return original(*args, **kwargs)

    server.policy.model.sample_actions = sample
    if not head_fault:
        running = noise_cases.RunningServer()
        running.http.RequestHandlerClass = noise_cases.server.make_handler(
            server, running.lock, noise_cases.OWNER
        )
        try:
            wire = noise_cases.server.pack({"request_id": 99, "task": "task", "captured_ns": 19}, arrays)
            code, _, reply = running.post(wire)
            assert code == 200
            header, output = noise_cases.server.unpack(reply)
            assert header["error"] is None and output["predicted_actions"].shape == (50, 32)
            assert torch.equal(torch.from_numpy(expected), torch.from_numpy(output["actions"]))
        finally:
            running.close()
    parked = server.residency("park")
    assert parked["state"] == "parked" and parked["cpu_vocabulary"]["cpu_storage_roots"] == 3
    resumed = server.residency("resume")
    assert resumed["state"] == "resident" and not resumed["cpu_vocabulary"]["faulted"]
    if head_fault:
        with pytest.raises(RuntimeError, match="outside the qualified action path"):
            server.chunk("task", arrays)
        assert server.residency_state == "fault"
        with pytest.raises(RuntimeError, match="not resident"):
            server.chunk("task", arrays)
    else:
        actual = server.chunk("task", arrays)
        assert torch.equal(torch.from_numpy(expected), torch.from_numpy(actual))
        assert torch.equal(torch.from_numpy(prediction), torch.from_numpy(server.last_policy_chunk))
        status = server.residency("status")
        assert status["cpu_vocabulary"]["gather_calls"] == 2
        assert status["cpu_vocabulary"]["guarded_head_calls"] == 0
    assert not torch.cuda.is_initialized()


def test_unprepared_owned_vocabulary_transition_faults_without_policy_conversion(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _ = real_policy_port(policy)
    server.residency_cpu_vocabulary = True
    cuda_observation_ports(monkeypatch)
    converted = []
    monkeypatch.setattr(server.policy, "to", lambda *_args, **_kwargs: converted.append(True))
    with pytest.raises(RuntimeError, match="prepare the owned CPU vocabulary shadow"):
        server.residency("park")
    assert not converted and server.residency_state == "fault" and not torch.cuda.is_initialized()


class ReviewedScaledEmbedding(torch.nn.Embedding):
    """Small CPU fixture of the reviewed deployment arithmetic, never a model."""

    def __init__(self) -> None:
        super().__init__(11, 4)
        self.register_buffer("embed_scale", torch.tensor(2.0), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


@pytest.fixture
def scaled_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.utils import trinity_vocab_placement as vocab

    monkeypatch.setattr(vocab, "_scaled_embedding_type", lambda: ReviewedScaledEmbedding)


def test_scaled_lookup_and_buffers_preserve_original_arithmetic_and_shadow_cycles(
    memory: None, scaled_type: None
) -> None:
    model = SmallTables()
    model.lookup = ReviewedScaledEmbedding().to(dtype=torch.bfloat16).eval()
    tokens = torch.tensor([[0, 2, 10, 7], [1, 3, 3, 9]])
    with torch.inference_mode():
        expected = model(tokens).clone()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    scale = model.lookup.embed_scale
    placement = place(model)
    assert model.lookup.embed_scale is scale
    assert placement.metadata()["embedding_kind"] == "gemma-scaled"
    shadow = ParameterShadow(
        model, torch, "cpu", 14 * 1024**3, park_retain_bytes=32, resident_cpu_names=placement.cpu_names
    )
    for _ in range(2):
        with torch.inference_mode():
            assert torch.equal(model(tokens), expected)
        shadow.park()
        placement.check(parked=True)
        shadow.restore()
        shadow.resumed()
        placement.check()
        assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
    assert placement.gather_calls == 2 and not torch.cuda.is_initialized()


@pytest.mark.parametrize("mutation", ["forward", "buffer_shape", "buffer_dtype", "extra_buffer"])
def test_changed_scaled_arithmetic_or_buffer_ownership_faults(
    memory: None, scaled_type: None, mutation: str
) -> None:
    model = SmallTables()
    model.lookup = ReviewedScaledEmbedding().to(dtype=torch.bfloat16).eval()
    placement = place(model)
    if mutation == "forward":
        model.lookup.forward = lambda *_: torch.zeros(1)
    elif mutation == "buffer_shape":
        model.lookup.embed_scale = torch.ones(2, dtype=torch.bfloat16)
    elif mutation == "buffer_dtype":
        model.lookup.embed_scale = model.lookup.embed_scale.float()
    else:
        model.lookup.register_buffer("extra", torch.zeros(1))
    with pytest.raises(RuntimeError), torch.inference_mode():
        model.lookup(torch.tensor([[0]]))
    assert placement.faulted and placement.gather_calls == 0


def test_unreviewed_scaled_forward_is_refused_before_conversion(
    memory: None, scaled_type: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = SmallTables()
    model.lookup = ReviewedScaledEmbedding().to(dtype=torch.bfloat16).eval()
    monkeypatch.setattr(ReviewedScaledEmbedding, "forward", lambda self, tokens: tokens)
    with pytest.raises(ValueError, match="qualified scaled lookup arithmetic"):
        place(model)
    assert not model.lookup._forward_pre_hooks


def test_instance_embedding_forward_override_is_refused_before_conversion(memory: None) -> None:
    model = SmallTables()
    model.lookup.forward = lambda *_: torch.zeros(1)
    with pytest.raises(ValueError, match="custom embedding forward"):
        place(model)
    assert not model.lookup._forward_pre_hooks


def test_lookup_execution_failure_faults_owner(memory: None) -> None:
    model = SmallTables()
    placement = place(model)
    with pytest.raises(IndexError), torch.inference_mode():
        model.lookup(torch.tensor([[99]]))
    assert placement.faulted and placement.gather_calls == 0


@pytest.mark.parametrize("vision_device", ["cpu", "meta"])
@pytest.mark.parametrize("video", [False, True])
@pytest.mark.parametrize("missing_camera", [False, True])
def test_real_image_preprocessor_follows_vision_device_instead_of_unused_output_head(
    vision_device: str, video: bool, missing_camera: bool
) -> None:
    """Meta observes routing only; no checkpoint, GPU or vision inference is run."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    model = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(model)
    key = "observation.images.image"
    model.config = SimpleNamespace(
        image_features={key: None, **({"missing": None} if missing_camera else {})},
        image_resolution=(8, 8),
    )
    model.model = torch.nn.Module()
    model.model.paligemma_with_expert = torch.nn.Module()
    pali = model.model.paligemma_with_expert.paligemma = torch.nn.Module()
    pali.lm_head = torch.nn.Linear(3, 3, bias=False)
    pali.model = torch.nn.Module()
    pali.model.vision_tower = torch.nn.Linear(3, 3, bias=False, device=vision_device)
    assert next(model.parameters()) is pali.lm_head.weight
    shape = (1, 2, 3, 8, 8) if video else (1, 3, 8, 8)
    source = torch.linspace(0, 1, torch.tensor(shape).prod().item()).reshape(shape)
    images, masks = model._preprocess_images({key: source})
    assert len(images) == len(masks) == (2 if missing_camera else 1)
    assert images[0].shape == source.shape and masks[0].shape == ((1, 2) if video else (1,))
    assert all(tensor.device.type == vision_device for tensor in [*images, *masks])
    if vision_device == "cpu":
        assert torch.equal(images[0], source * 2 - 1) and masks[0].all()
        if missing_camera:
            assert torch.equal(images[1], torch.ones_like(source) * -1) and not masks[1].any()
    assert not torch.cuda.is_initialized()


def test_owned_vocabulary_inference_error_preserves_traceback_and_wire_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    running = noise_cases.RunningServer()
    running.model.residency_cpu_vocabulary = True

    def fail(*_args: Any) -> Any:
        raise RuntimeError("original mixed-device inference error")

    running.model.chunk = fail
    try:
        wire = noise_cases.server.pack({"request_id": 17, "task": "task", "captured_ns": 19}, {})
        code, headers, reply = running.post(wire)
        header, arrays = noise_cases.server.unpack(reply)
        assert code == 409 and headers["X-Trinity-Owner"] == noise_cases.OWNER
        assert header["error"] == "RuntimeError: original mixed-device inference error" and not arrays
    finally:
        running.close()
    assert "original mixed-device inference error" in capsys.readouterr().err


def test_fixed_cpu_table_storage_is_not_replaced_during_shadow_parking(memory: None) -> None:
    model = SmallTables()
    placement = place(model)
    before = {
        name: dict(model.named_parameters())[name].untyped_storage().data_ptr()
        for name in placement.cpu_names
    }
    shadow = ParameterShadow(
        model, torch, "cpu", 14 * 1024**3, park_retain_bytes=32, resident_cpu_names=placement.cpu_names
    )
    for _ in range(2):
        shadow.park()
        assert before == {
            name: dict(model.named_parameters())[name].untyped_storage().data_ptr()
            for name in placement.cpu_names
        }
        shadow.restore()
        shadow.resumed()
        placement.check()
    assert not torch.cuda.is_initialized()
