# Pinned, directly source-derived GENCO runtime

Upstream: <https://github.com/gridfm/gridfm-graphkit>, commit
`0912c734b0912202e22aad79dde996ae0f1190a0` (Apache-2.0).

`pristine/` contains byte-for-byte public files retrieved using immutable raw
commit URLs. It includes the full original `param_handler.py`, original package
initializer, computational files, license, README, preprocessing/normalization
references, and the tiny/case30 recipes. `PROVENANCE.json` hashes each file and
records the adaptations. **Do not describe `runtime/` as unmodified upstream.**

The model, physics decoders, masking and losses are the official computational
source, not an inspired reimplementation. The only adaptations are:

* Lightweight package/subpackage initializers avoid eager task/Lightning imports.
* `io/param_handler.py` keeps the exact upstream `get_physics_decoder` function
  (original lines 155–164), plus the two imports it needs.
* Three computational modules replace only `torch_scatter.scatter_add` imports
  with a disclosed `_scatter.py` compatibility import and add an adaptation notice.
  Importable `torch_scatter` remains the default. If its import fails with
  ImportError/OSError (including missing binary extensions), the fallback uses
  PyG's **native sum** implementation. It supports only the audited 1-D index
  calls. `GENCO_SCATTER_BACKEND=pyg_native_sum` explicitly selects this backend.
* The editor normalizes the final blank line in the adapted loss file; there is
  no additional computational change.

No upstream setup script or project requirements were executed. No Torch/NumPy
replacement or new dependency is required. Upstream's declared newer dependency
floor is not claimed here: this bounded CPU integration is tested separately
against Torch 2.1.2+cu121 / PyG 2.6.1 / NumPy 1.26.3.

## Licensing

Keep `LICENSE`, `pristine/LICENSE`, all original source notices, this adaptation
notice, and `PROVENANCE.json` with redistribution. No upstream root `NOTICE` file
exists at this pin (verified against the repository root inventory). This README
is a local provenance/adaptation notice, not a purported upstream NOTICE.

## Integrity

`research.genco_pf_baseline_20260921.runtime.verify_vendor()` checks every
manifested file; `construct_preset()` performs this check and verifies imported
module origins. The tests also establish that the three adapted computations
differ only by their disclosed import/notice, and that other computational files
remain byte-identical.
