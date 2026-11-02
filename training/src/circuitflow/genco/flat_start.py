"""A from-scratch GENCO initialization ablation, not a new architecture or SOTA claim."""
import torch

from circuitflow.genco.runtime import Runtime, construct_preset


def construct_flat_start_preset(preset: str) -> Runtime:
    """Center the shared [Vm, angle] head at [1 pu, 0 rad]; preserve all other draws."""
    result = construct_preset(preset)
    head = result.model.get_submodule("mlp_bus.3")
    if not isinstance(head, torch.nn.Linear) or head.bias is None or head.bias.shape != (2,):
        raise ValueError("Pinned GENCO shared [Vm, angle] head structure changed")
    previous = head.bias.detach().tolist()
    with torch.no_grad():
        head.bias.copy_(head.bias.new_tensor([1., 0.]))
    result.metadata = {
        **result.metadata,
        "name": result.metadata["name"].replace("-PF-MASKALIGNED-", "-FSI-PF-MASKALIGNED-", 1),
        "initialization_ablation": {
            "changed_parameter": "mlp_bus.3.bias",
            "original_bias": previous, "new_bias": [1., 0.],
            "units": ["pu_voltage_magnitude", "radian_voltage_angle"],
            "parameter_count_changed": False, "additional_rng_draws": 0,
            "head_weights_zeroed": False, "known_control_masks_changed": False,
            "loss_or_feedback_changed": False,
            "positive_voltage_guarantee": False,
            "prior_art": "https://arxiv.org/html/2609.16738v1#S4",
            "new_architecture_or_novelty_claim": False,
            "trained_or_quality_validated": False,
        },
    }
    return result


def construct_flat_start_tiny() -> Runtime:
    return construct_flat_start_preset("tiny")
