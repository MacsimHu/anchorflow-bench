"""Local compatibility adaptation, not an upstream GraphKit file.

Only the audited one-dimensional-index sum calls are supported by the fallback.
An installed, importable torch_scatter remains the default. No packages are
installed here. Set GENCO_SCATTER_BACKEND=pyg_native_sum before first import to
exercise the native implementation explicitly.
"""
import os

from torch_geometric.utils import scatter


def native_scatter_add(src, index, dim=-1, out=None, dim_size=None):
    if out is not None:
        raise ValueError("The GENCO native fallback does not support out=")
    if index.ndim != 1:
        raise ValueError("The GENCO native fallback requires a 1-D index")
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


requested = os.environ.get("GENCO_SCATTER_BACKEND", "auto")
if requested not in ("auto", "torch_scatter", "pyg_native_sum"):
    raise ValueError(f"Unknown GENCO_SCATTER_BACKEND: {requested}")
IMPORT_ERROR = None
if requested == "pyg_native_sum":
    scatter_add = native_scatter_add
    BACKEND = "pyg_native_sum"
else:
    try:
        from torch_scatter import scatter_add
        BACKEND = "torch_scatter"
    except (ImportError, OSError) as exc:
        if requested == "torch_scatter":
            raise
        IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        scatter_add = native_scatter_add
        BACKEND = "pyg_native_sum"
