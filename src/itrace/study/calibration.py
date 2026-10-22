from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from .jsonio import JsonValue, StudyError, fields, mapping, number
from .model import Limits
from .post_validation import compare_validation
from .setup import Setup, rectangle

FIT: Final = tuple((x, y) for y in (0.15, 0.5, 0.85) for x in (0.15, 0.5, 0.85))
HELDOUT: Final = ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7), (0.5, 0.35))


@dataclass(frozen=True, slots=True)
class Frame:
    timestamp_s: float
    epoch: str
    gaze_x: float
    gaze_y: float
    valid: bool


@dataclass(frozen=True, slots=True)
class TargetOnset:
    estimate_s: float
    uncertainty_s: float
    x: float
    y: float
    epoch: str


@dataclass(frozen=True, slots=True)
class Dwell:
    gaze_x: float
    gaze_y: float
    x: float
    y: float
    payload: bytes


class Calibration:
    def __init__(self, setup: Setup, limits: Limits) -> None:
        self.setup = setup
        self.limits = limits
        self.index = 0
        self.onset: TargetOnset | None = None
        self.fit: list[Dwell] = []
        self.heldout: list[Dwell] = []
        self.post: list[Dwell] = []
        self.pre_residuals: tuple[tuple[float, float], ...] = ()
        self.coefficients: tuple[tuple[float, ...], ...] | None = None
        self.diagnostics: dict[str, JsonValue] = {}

    def target(self) -> dict[str, JsonValue] | None:
        if self.index >= 19:
            return None
        x, y = (*FIT, *HELDOUT, *HELDOUT)[self.index]
        return {
            "target_id": f"target-{self.index + 1}",
            "phase": "calibration"
            if self.index < 9
            else "validation"
            if self.index < 14
            else "post_validation",
            "x_fraction": x,
            "y_fraction": y,
            "settle_s": self.limits.settle_s,
            "collect_s": self.limits.collect_s,
        }

    def begin(
        self, payload: Mapping[str, JsonValue], estimate: float, uncertainty: float, epoch: str
    ) -> TargetOnset:
        target = self.target()
        if target is None or self.onset is not None or payload["target_id"] != target["target_id"]:
            raise StudyError("target_id", "unexpected target or duplicate onset")
        rect = rectangle(payload["rect"], self.setup)
        position = mapping(payload["position"], "position")
        fields(position, "x y")
        x = number(position["x"], "position.x", 0, self.setup.width)
        y = number(position["y"], "position.y", 0, self.setup.height)
        fx, fy = (*FIT, *HELDOUT, *HELDOUT)[self.index]
        if abs(x - fx * self.setup.width) > 2 or abs(y - fy * self.setup.height) > 2:
            raise StudyError("position", "target must match full viewport CSS coordinates")
        if not (
            number(rect["x"], "x") <= x <= number(rect["x"], "x") + number(rect["width"], "w")
            and number(rect["y"], "y") <= y <= number(rect["y"], "y") + number(rect["height"], "h")
        ):
            raise StudyError("position", "target center outside actual rectangle")
        return TargetOnset(estimate, uncertainty, x, y, epoch)

    def collect(
        self,
        payload: Mapping[str, JsonValue],
        estimate: float,
        uncertainty: float,
        frames: Sequence[Frame],
        epoch: str,
    ) -> Dwell:
        from .jsonio import encoded

        target, onset = self.target(), self.onset
        if target is None or onset is None or payload["target_id"] != target["target_id"]:
            raise StudyError("target_id", "target onset required")
        start = onset.estimate_s + onset.uncertainty_s + self.limits.settle_s
        end = start + self.limits.collect_s
        if epoch != onset.epoch or estimate - uncertainty < end:
            raise StudyError("target_collect", "capture epoch changed or collection too early")
        selected = [
            frame for frame in frames if frame.epoch == epoch and start <= frame.timestamp_s < end
        ]
        valid = [frame for frame in selected if frame.valid]
        fraction = len(valid) / len(selected) if selected else 0.0
        adequate = (
            len(valid) >= self.limits.min_samples and fraction >= self.limits.min_valid_fraction
        )
        if not adequate and self.index < 14:
            raise StudyError(
                "calibration", "insufficient finite post-settle samples or valid fraction"
            )
        x = math.fsum(frame.gaze_x for frame in valid) / len(valid) if adequate else math.nan
        y = math.fsum(frame.gaze_y for frame in valid) / len(valid) if adequate else math.nan
        return Dwell(
            x,
            y,
            onset.x,
            onset.y,
            encoded(
                {
                    "target_id": target["target_id"],
                    "phase": target["phase"],
                    "window_start_s": start,
                    "window_end_s": end,
                    "epoch_id": epoch,
                    "gaze_x_deg": x if adequate else None,
                    "gaze_y_deg": y if adequate else None,
                    "quality_adequate": adequate,
                    "target_x_css_px": onset.x,
                    "target_y_css_px": onset.y,
                    "total_samples": len(selected),
                    "valid_samples": len(valid),
                    "valid_fraction": fraction,
                    "excluded_samples": len(selected) - len(valid),
                    "dropout_fraction": 1 - fraction,
                }
            ),
        )

    def accept(self, dwell: Dwell) -> None:
        if self.index < 9:
            self.fit.append(dwell)
        elif self.index < 14:
            self.heldout.append(dwell)
        else:
            self.post.append(dwell)
        self.index += 1
        self.onset = None
        if self.index == 9:
            design = np.array([[point.gaze_x, point.gaze_y, 1] for point in self.fit])
            targets = np.array([[point.x, point.y] for point in self.fit])
            try:
                coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
            except np.linalg.LinAlgError as exc:
                raise StudyError("calibration", "affine least-squares failed") from exc
            if rank != 3 or not np.all(np.isfinite(coefficients)):
                raise StudyError("calibration", "rank-deficient affine fit")
            self.coefficients = tuple(tuple(float(v) for v in row) for row in coefficients)
            residuals = np.linalg.norm(design @ coefficients - targets, axis=1)
            self.diagnostics = {
                "fit_rms_css_px": float(np.sqrt(np.mean(residuals**2))),
                "fit_rank": int(rank),
                "fit_target_count": 9,
                "coefficients_gaze_deg_to_css_px": [list(row) for row in self.coefficients],
                "coordinate_units": "native gaze degrees -> viewport CSS pixels",
            }
        if self.index == 14:
            if self.coefficients is None:
                raise StudyError("calibration", "affine fit unavailable")
            coefficients = np.array(self.coefficients)
            design = np.array([[p.gaze_x, p.gaze_y, 1] for p in self.heldout])
            targets = np.array([[p.x, p.y] for p in self.heldout])
            residuals = design @ coefficients - targets
            self.pre_residuals = tuple((float(row[0]), float(row[1])) for row in residuals)
            errors = np.linalg.norm(residuals, axis=1)
            diagonal = math.hypot(self.setup.width, self.setup.height)
            rms = float(np.sqrt(np.mean(errors**2)))
            p95 = float(np.percentile(errors, 95))
            self.diagnostics.update(
                {
                    "heldout_target_count": 5,
                    "heldout_rms_css_px": rms,
                    "heldout_p95_css_px": p95,
                    "heldout_rms_fraction": rms / diagonal,
                    "heldout_p95_fraction": p95 / diagonal,
                    "last_validation_error_css_px": float(errors[-1]),
                    "no_device_accuracy_claim": True,
                }
            )
            if not math.isfinite(p95) or p95 / diagonal > self.limits.max_validation_error_fraction:
                raise StudyError(
                    "validation", "heldout error exceeds declared screen-diagonal fraction"
                )
        if 14 < self.index <= 19 and self.coefficients is not None:
            self.diagnostics.update(
                compare_validation(
                    self.coefficients,
                    self.pre_residuals,
                    [(point.gaze_x, point.gaze_y, point.x, point.y) for point in self.post],
                    math.hypot(self.setup.width, self.setup.height),
                    self.limits.max_validation_error_fraction,
                )
            )
