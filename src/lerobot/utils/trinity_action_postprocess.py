"""Guard chunk batching for the stateless pi0.5 LIBERO output pipeline."""

from __future__ import annotations

from typing import Any


def check_batch_postprocess(config: Any, postprocessor: Any, env_postprocessor: Any) -> None:
    """Refuse batch-dependent or custom output processors before model inference."""
    from lerobot.configs.types import FeatureType, NormalizationMode
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        DeviceProcessorStep,
        UnnormalizerProcessorStep,
    )
    from lerobot.processor.converters import (
        batch_to_transition,
        policy_action_to_transition,
        transition_to_batch,
        transition_to_policy_action,
    )
    from lerobot.processor.pipeline import DataProcessorPipeline
    from lerobot.utils.constants import ACTION

    if getattr(config, "type", None) != "pi05" or getattr(config, "use_relative_actions", False):
        raise ValueError("batch action postprocessing requires stateless pi05 actions")
    if (
        type(postprocessor) is not DataProcessorPipeline
        or type(env_postprocessor) is not DataProcessorPipeline
    ):
        raise ValueError("batch action postprocessing requires the qualified processor pipelines")
    for pipeline in (postprocessor, env_postprocessor):
        if pipeline.before_step_hooks or pipeline.after_step_hooks:
            raise ValueError("batch action postprocessing does not support processor hooks")
    if (
        postprocessor.to_transition is not policy_action_to_transition
        or postprocessor.to_output is not transition_to_policy_action
        or env_postprocessor.to_transition is not batch_to_transition
        or env_postprocessor.to_output is not transition_to_batch
        or env_postprocessor.steps
    ):
        raise ValueError("batch action postprocessing requires unchanged output converters and environment")
    steps = postprocessor.steps
    if len(steps) not in (2, 3) or type(steps[0]) is not UnnormalizerProcessorStep:
        raise ValueError("batch action postprocessing requires the qualified unnormalizer")
    normalizer = steps[0]
    feature = normalizer.features.get(ACTION)
    stats = normalizer._tensor_stats.get(ACTION, {})
    if (
        set(normalizer.features) != {ACTION}
        or feature is None
        or feature.type is not FeatureType.ACTION
        or tuple(feature.shape) != (7,)
        or normalizer.norm_map.get(FeatureType.ACTION) is not NormalizationMode.MEAN_STD
        or any(name not in stats or tuple(stats[name].shape) != (7,) for name in ("mean", "std"))
    ):
        raise ValueError("batch action postprocessing requires seven-channel mean/std statistics")
    if len(steps) == 3 and (type(steps[1]) is not AbsoluteActionsProcessorStep or steps[1].enabled):
        raise ValueError("batch action postprocessing does not support relative actions")
    last = steps[-1]
    if (
        type(last) is not DeviceProcessorStep
        or last.tensor_device.type != "cpu"
        or last.float_dtype is not None
    ):
        raise ValueError("batch action postprocessing requires the unchanged CPU output step")
