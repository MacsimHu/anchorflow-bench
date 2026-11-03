"""Model construction and inference without a benchmark checkout or training labels."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .genco.adapter import PFAdapter
from .genco.flat_start import construct_flat_start_preset
from .genco.normalization import PowerScale
from .genco.runtime import Runtime, construct_preset
from .model import construct_circuit_preset


@dataclass(frozen=True)
class ModelSpec:
    name: str = "circuitflow"
    coupling: str = "coupled"
    coarse_weighting: str = "electrical"
    max_component_buses: int = 512

    def __post_init__(self):
        if self.name not in ("circuitflow", "genco_h48", "genco_h48_fsi"):
            raise ValueError("Unknown model recipe")
        if self.coupling not in ("coupled", "deferred"):
            raise ValueError("Unknown feedback control")
        if self.coarse_weighting not in ("electrical", "unit", "mean_token", "knockout"):
            raise ValueError("Unknown harmonic-basis control")
        if type(self.max_component_buses) is not int or self.max_component_buses < 1:
            raise ValueError("A positive component-size safety limit is required")
        if self.name != "circuitflow" and (
                self.coupling != "coupled" or self.coarse_weighting != "electrical"):
            raise ValueError("Feedback/coarse-space controls apply only to CircuitFlow")


def construct(spec: ModelSpec, device="cpu") -> Runtime:
    if spec.name == "circuitflow":
        runtime = construct_circuit_preset(
            "tiny", coupling=spec.coupling, coarse_weighting=spec.coarse_weighting,
            max_component_buses=spec.max_component_buses)
    elif spec.name == "genco_h48":
        runtime = construct_preset("case30_yaml")
    else:
        runtime = construct_flat_start_preset("case30_yaml")
    runtime.model.to(device)
    return runtime


def forward(runtime, adapter, raw, device, *, need_loss=True):
    inputs = adapter.build_inputs(raw).to(device)
    targets = ({key: value.to(device) for key, value in adapter.build_targets(raw).items()}
               if need_loss else None)
    predicted = runtime.model(
        inputs.x_dict, inputs.edge_index_dict, inputs.edge_attr_dict, inputs.mask_dict)
    loss = (runtime.criterion(
        predicted, targets, inputs.edge_index_dict, inputs.edge_attr_dict, inputs.mask_dict,
        model=runtime.model, x_dict=inputs.x_dict)["loss"] if need_loss else None)
    outputs = adapter.decode(predicted, raw.to(device))
    if any(not torch.isfinite(value).all() for value in outputs.values() if torch.is_tensor(value)):
        raise FloatingPointError("The model returned a nonfinite physical output")
    if loss is not None and not torch.isfinite(loss):
        raise FloatingPointError("The training objective is nonfinite")
    return outputs, loss


def load_inference(path, *, device="cpu"):
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved.get("schema") != "circuitflow-inference-v1":
        raise ValueError("Use a CircuitFlow inference export, not an arbitrary training pickle")
    spec = ModelSpec(**saved["model_spec"])
    runtime = construct(spec, device)
    runtime.model.load_state_dict(saved["model"], strict=True)
    runtime.model.eval().requires_grad_(False)
    scale = PowerScale.from_dict(saved["power_scale"])
    adapter = PFAdapter(power_scale_pu=scale, generator_layout="aggregate_pf")
    return runtime, adapter, saved["provenance"]


def inference_payload(runtime, spec, scale, provenance):
    return dict(schema="circuitflow-inference-v1", model_spec=asdict(spec),
                model={key: value.detach().cpu() if torch.is_tensor(value) else value
                       for key, value in runtime.model.state_dict().items()},
                power_scale=scale.to_dict(), provenance=provenance)
