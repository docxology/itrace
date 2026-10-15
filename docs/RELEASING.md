# Releasing iTrace

A release binds the software, manuscript, evidence, and version DOI to one Git
commit. Local gates, hosted CI, and publication are separate results. A failed
gate blocks publication and remains part of the evidence.

## Prepare

1. Verify the public repository identity, mounted development volume, runtime,
   current owners, dirty files, and remote branch. Preserve unrelated work.
2. Choose the next version. For a Zenodo update, create a new version of the
   existing concept, reserve its DOI, and retain the draft ID before any retry.
   Preserve the previous published version.
3. Update `src/itrace/version.py`, `pyproject.toml`, the root package entry in
   `uv.lock`, `CITATION.cff`, `.zenodo.json`, the current verification ledger and
   traceability matrix, README, abstract, cover date, and versioned release notes.
   Keep historical release notes and evidence unchanged. `CITATION.cff` cites
   the version DOI and also lists the all-version concept DOI.
4. Run the full development suite and the core-only suite, recording actual
   counts, coverage, skips, and runtime failures. Refresh the ledger from those
   results. Regenerate figures and manuscript; inspect their actual output.

The explicit retained-summary figure mode is appropriate when historical derived
session exports are unavailable:

```bash
uv run --frozen python scripts/generate_figures.py --reuse-empirical-summaries
uv run --frozen python scripts/render_manuscript.py
uv run --frozen python scripts/verify.py --out-dir output/verification/release-new
```

`scripts/audit_release.py`, also invoked by `verify.py`, rejects mismatched
versions, dates, DOIs, missing release notes, empty/duplicate/escaped figure
inventories, modified figure bytes, and invalid PDF headers. These checks do
not replace visual inspection, installed-package checks, or scientific review.

## Freeze and build

Commit only reviewed, intended changes. Leave local captures, caches, draft
credentials, task scratch files, and unrelated render exports out of Git and the
release. Record the exact commit, then build both distributions from that
commit's isolated archive:

```bash
uv build --out-dir "$RELEASE_ASSETS"
uv venv "$WHEEL_ENV"
uv pip install --python "$WHEEL_ENV/bin/python" "$RELEASE_ASSETS"/itrace-*.whl
"$WHEEL_ENV/bin/python" -I -c 'import itrace; print(itrace.__version__, itrace.__file__)'
"$WHEEL_ENV/bin/itrace" demo
```

Use external volume/cache/TMP/runtime bindings where the project's environment
requires them. Smoke-test the sdist in another environment and inspect package
assets. CI exercises Python 3.10-3.13 on Linux and an installed wheel with optional
dependencies absent. The development lane checks coverage, lint, format, typing,
publication consistency, and figure hashes.

Include the source archive, wheel, sdist, manuscript, current verification and
figure ledgers, historical empirical summary ledgers, notes, license, and citation
in a fresh asset directory. Generate an inventory after all payloads are final:

```python
from pathlib import Path
from itrace.release import write_asset_manifest

write_asset_manifest(Path(asset_directory), exact_commit_sha)
```

The inventory refuses empty files, symlinks, subdirectories, ambiguous filenames,
abbreviated commits, and existing inventory files. `SHA256SUMS` covers payload
files; `release-manifest.json` also binds the checksum file and exact source
commit. Neither checksum receipt attempts to hash itself.

## Publish and verify

Push `main` without rewriting history. Require hosted checks on the exact release
commit to pass, then create its version tag and GitHub release with the reviewed
notes and complete asset set. Upload the identical asset set to the reserved
Zenodo draft. Check every remote filename, length, and checksum before publishing.
Replace inherited files only within that new unpublished draft, preserving the
previous published version.

After publication, download assets anonymously from both hosts and compare their
SHA-256 values against the local inventory, including the manifest itself. Verify
the tag/commit, GitHub release, version record, concept relationship, metadata,
and DOI links. Keep publication receipts outside the immutable release payload.
Record a DOI redirect that is still propagating rather than claiming it resolves.
Preserve frozen assets and tags if publication needs repair.

The release does not close roadmap items requiring new participant data,
reference instruments, dataset access, or distinct method design and validation.
Those remain explicit in `TODO.md`.
