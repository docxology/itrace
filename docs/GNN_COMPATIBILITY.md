# Running GNN alongside iTrace

iTrace acquires and analyzes derived gaze records. It does not import GNN or
render generative models. Keep a GNN installation separate from a webcam study
environment; adding GNN to that environment does not connect it to the study.

## Dependency boundary

The current [GNN package](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/pyproject.toml)
requires NumPy 2 or later. iTrace's pinned MediaPipe 0.10.21 capture backend
requires NumPy below 2. These requirements cannot be satisfied in one Python
environment. The iTrace core supports other NumPy versions, but capture has
this additional constraint. Do not relax either package's declared requirements
or replace the capture backend merely to combine installations.

Use each repository's committed lockfile and its own environment. Keep the
runtime and base interpreter on the intended storage volume, and bind
`UV_PROJECT_ENVIRONMENT`, `UV_PYTHON`, caches and temporary directories explicitly.
GNN's `uv.toml` may also require an exact uv version; follow that requirement
rather than rewriting it to match a workstation's default tool.

For GNN, compare both package and source versions in its selected environment:

```bash
uv run --locked --no-sync python -c \
  'import gnn, importlib.metadata as m; print(gnn.__version__); print(m.version("generalized-notation-notation")); print(gnn.__file__)'
uv run --locked --no-sync gnn --help
```

An editable checkout can expose current source while retaining old distribution
metadata. Refresh its locked installation and verify both values before using
it. Keep an environment owned by a live process unchanged; create a separate
environment for the refresh. Consult the [GNN version map](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/VERSION_MAP.md)
and [published releases](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/releases)
for current software and release identities.

## Study and model evidence

Treat any later model integration as a separate, versioned analysis boundary.
Bind the GNN revision, model bytes and parser/extractor identity, as well as the
iTrace session manifest, captured records and declared observation mapping.
Do not resume scored collection, rewrite old provenance or treat a synthetic
model execution as participant behavior. Parsing a model does not establish that
it describes a recorded session or that its predictions fit the participant.

For collection, follow the [self-experiment run guide](RUN_SELF_EXPERIMENT.md).
For software verification and installed-package checks, follow
[development and verification](DEVELOPMENT.md).
