# Portable training source

This source-only Apache-2.0 package preserves the previously qualified
CircuitFlow (paper name AnchorFlow) and pinned GENCO implementation.
The benchmark parser retains its MIT notice. See `SOURCE_PROVENANCE.json`,
`EXPORT_MANIFEST.json`, `LICENSE` and `licenses/`.

No example weights, datasets, collaborator assignments, private machine
configuration or manuscripts are included. Install from the repository root
using `python -m pip install --no-deps -e ./training`, after installing
`training/requirements-reproduce.txt` in a separate Python 3.10 environment.

Use the root `anchortrain.py` company entry point. The legacy package CLI
deliberately retains its original collaboration-job restrictions and must not
be edited to bypass them. The company wrapper creates its own protocol and
identity. The original model, loss, batch schedule, optimizer and H200 endpoint
are unchanged. An exported checkpoint is not a new scientific replicate.

`python -m pytest training/tests` checks model API, label isolation, data
preparation and complete model/Adam/scheduler/RNG/cursor restoration.
These checks do not certify another machine's GPU or benchmark accuracy.
