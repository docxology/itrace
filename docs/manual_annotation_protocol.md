# Manual Annotation Evidence Protocol

Manual annotation is an evidence lane for the repeated-session v1 gate. It is
not a reference eye-tracker comparison and does not make population,
cross-device, or device-accuracy claims by itself.

## Artifact Contract

Attach a repo-relative `reference_artifact` JSON path to the chosen
`docs/empirical_sessions_manifest.json` session row. The JSON must be derived
from the session report and capture-record CSV only; raw video and persisted
eye-crop images are outside the default workflow.

Required JSON fields:

- `kind`: `itrace_manual_annotation_evidence`
- `version`: numeric schema version
- `session_id`: matching manifest `session_id`
- `source_report`: repo-relative `experiment_report.json`
- `source_records`: repo-relative derived capture-record CSV
- `annotation_scope`: normally `prompted_target_windows`
- `annotator_id`: pseudonymous annotator label, such as `A001`
- `created_at`: timestamp string
- `annotations`: nonempty list of sampled target-window judgments

Each annotation row needs `trial_id`, `target_label`, numeric `start_s` and
`end_s`, `quality` as `usable`, `exclude`, or `uncertain`, and `target_hit` as
`yes`, `no`, or `uncertain`.

## Template

```json
{
  "kind": "itrace_manual_annotation_evidence",
  "version": 1,
  "session_id": "local_pilot_003",
  "source_report": "output/empirical_pilot/local_pilot_003/experiment/experiment_report.json",
  "source_records": "output/empirical_pilot/local_pilot_003/experiment/trial_corner_saccades_capture_records.csv",
  "annotation_scope": "prompted_target_windows",
  "annotator_id": "A001",
  "created_at": "2026-06-08T12:00:00Z",
  "annotations": [
    {
      "trial_id": "corner_saccades",
      "target_label": "center",
      "start_s": 0.0,
      "end_s": 1.5,
      "quality": "usable",
      "target_hit": "yes"
    }
  ]
}
```
