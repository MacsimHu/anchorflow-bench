"""Load the pinned, minimally adapted official GENCO computation, never a clone."""
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch


SOURCE_PIN = "0912c734b0912202e22aad79dde996ae0f1190a0"
VENDOR = Path(__file__).resolve().parents[1] / "_vendor" / "genco"
RUNTIME = VENDOR / "runtime"
PRIMARY_MASK = "REFVmKnown"
PAPER_MASK = "REFVmHidden"
PRESETS = {"tiny": (12, 1276611), "small": (24, 5047011), "case30_yaml": (48, 20069187)}


def _inside_runtime(module):
    filename = getattr(module, "__file__", None)
    return filename is not None and Path(filename).resolve().is_relative_to(RUNTIME)


def load_official():
    """Fail closed if another GraphKit package already occupies the namespace."""
    for name, module in tuple(sys.modules.items()):
        if name == "gridfm_graphkit" or name.startswith("gridfm_graphkit."):
            if not _inside_runtime(module):
                raise ImportError(f"Non-pinned GraphKit module already loaded: {name}")
    location = str(RUNTIME)
    if location in sys.path:
        sys.path.remove(location)
    sys.path.insert(0, location)
    model = importlib.import_module("gridfm_graphkit.models.gnn_heterogeneous_gns")
    loss = importlib.import_module("gridfm_graphkit.training.loss")
    for name, module in tuple(sys.modules.items()):
        if name == "gridfm_graphkit" or name.startswith("gridfm_graphkit."):
            if not _inside_runtime(module):
                raise ImportError(f"Import escaped pinned vendor path: {name}")
    return model, loss


def verify_vendor():
    manifest = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
    if manifest["source_commit"] != SOURCE_PIN:
        raise RuntimeError("Unexpected GENCO source pin")
    for entry in manifest["files"]:
        path = VENDOR / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Vendor integrity check failed: {entry['path']}")
    return hashlib.sha256((VENDOR / "PROVENANCE.json").read_bytes()).hexdigest()


@dataclass
class Runtime:
    model: torch.nn.Module
    criterion: torch.nn.Module
    args: NS
    metadata: dict


def construct_preset(preset="tiny", *, mask=PRIMARY_MASK, dtype=torch.float32):
    """CPU construction only; return the exact upstream model, loss, args, metadata."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {tuple(PRESETS)}")
    if mask not in (PRIMARY_MASK, PAPER_MASK):
        raise ValueError("mask must be REFVmKnown or REFVmHidden")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Supported precision: float32 or float64")
    manifest_sha = verify_vendor()
    official, losses = load_official()
    hidden, expected = PRESETS[preset]
    args = NS(
        task=NS(task_name="PowerFlow"),
        model=NS(num_layers=12, hidden_size=hidden, attention_head=8,
                 input_bus_dim=15, input_gen_dim=6, edge_dim=10,
                 output_bus_dim=2, output_gen_dim=1, dropout=0.0),
    )
    model = official.GNS_heterogeneous(args).to(device="cpu", dtype=dtype)
    criterion = losses.MixedLoss(
        [losses.LayeredWeightedPhysicsLoss(NS(base_weight=0.5), args),
         losses.MaskedBusMSE(NS(), args)],
        weights=[0.1, 0.9],
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected:
        raise RuntimeError(f"Upstream/PyG parameter count changed: {count} != {expected}")
    backend = importlib.import_module("gridfm_graphkit._scatter")
    import torch_geometric
    name = {"tiny": "Tiny", "small": "Small", "case30_yaml": "Case30YAML"}[preset]
    variant = "PF-MASKALIGNED-REFVmKnown" if mask == PRIMARY_MASK else "PAPER-MASK-REFVmHidden"
    metadata = {
        "name": f"GENCO-{name}-{variant}", "source_commit": SOURCE_PIN,
        "source_repository": "https://github.com/gridfm/gridfm-graphkit",
        "manifest_sha256": manifest_sha,
        "model_module": str(Path(official.__file__).resolve()),
        "preset": preset, "mask": mask, "precision": str(dtype),
        "scatter_backend": backend.BACKEND, "scatter_import_error": backend.IMPORT_ERROR,
        "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
        "parameter_count": count, "device_at_construction": "cpu",
        "objective": {"LayeredWeightedPhysicsLoss": 0.1, "base_weight": 0.5, "MaskedBusMSE": 0.9},
        "official_case30_recipe_reference": {
            "optimizer": "AdamW", "learning_rate": 5e-4, "betas": [0.9, 0.999],
            "weight_decay": 0.01, "eps": 1e-8,
            "scheduler": "ReduceLROnPlateau", "factor": 0.7, "patience": 5,
            "batch_size": 64, "epochs": 200, "seed": 0,
            "applied_by_this_factory": False,
        },
        "adapted_runtime": True, "paper_dataset_or_result_replication": False,
        "physics_feedback_in_eval": True, "normalization": "explicit frozen PowerScale or caller-frozen power_scale_pu required",
    }
    return Runtime(model, criterion, args, metadata)


def construct_tiny(**kwargs):
    return construct_preset("tiny", **kwargs)
