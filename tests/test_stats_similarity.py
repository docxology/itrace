"""Tests for itrace.stats.similarity (no mocks; real data + hand-checked truth)."""

from __future__ import annotations

import numpy as np
import pytest

from itrace.encoding import encode_directions
from itrace.stats import similarity as sim
from itrace.synthetic import gaze_with_saccade
from itrace.types import GazeStream, Saccade


def _sacc(direction: float, amplitude: float) -> Saccade:
    return Saccade(0, 1, 0.0, 0.02, amplitude, direction, 300.0)


def _multi_saccade_stream(seed: int = 0) -> GazeStream:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    t_offset = 0.0
    for amp in np.linspace(2.0, 22.0, 24):
        direction = float(rng.uniform(-180.0, 180.0))
        g, _ = gaze_with_saccade(amplitude_deg=float(amp), direction_deg=direction, fixation_s=0.05)
        xs.append(g.x)
        ys.append(g.y)
        ts.append(g.t + t_offset)
        t_offset = float(ts[-1][-1]) + 1.0 / 250.0
    return GazeStream(t=np.concatenate(ts), x=np.concatenate(xs), y=np.concatenate(ys))


# --- levenshtein -----------------------------------------------------------


def test_levenshtein_identical_is_zero() -> None:
    assert sim.levenshtein("RULD", "RULD") == 0


def test_levenshtein_empty_inputs() -> None:
    assert sim.levenshtein("", "") == 0
    assert sim.levenshtein("abc", "") == 3
    assert sim.levenshtein("", "abcd") == 4


def test_levenshtein_single_substitution() -> None:
    # One differing character -> distance 1.
    assert sim.levenshtein("RULD", "RUXD") == 1


def test_levenshtein_classic_kitten_sitting() -> None:
    # Textbook example: kitten -> sitting needs 3 edits.
    assert sim.levenshtein("kitten", "sitting") == 3


def test_levenshtein_insertion_and_deletion() -> None:
    assert sim.levenshtein("RU", "RUL") == 1  # one insertion
    assert sim.levenshtein("RUL", "RU") == 1  # one deletion


def test_levenshtein_is_symmetric() -> None:
    assert sim.levenshtein("RRLLD", "RLDUU") == sim.levenshtein("RLDUU", "RRLLD")


# --- normalized_levenshtein ------------------------------------------------


def test_normalized_levenshtein_identical_is_zero() -> None:
    assert sim.normalized_levenshtein("RULD", "RULD") == 0.0


def test_normalized_levenshtein_both_empty_is_zero() -> None:
    assert sim.normalized_levenshtein("", "") == 0.0


def test_normalized_levenshtein_one_of_four_changed() -> None:
    # Distance 1 over max length 4 -> 0.25.
    assert sim.normalized_levenshtein("RULD", "RUXD") == pytest.approx(0.25)


def test_normalized_levenshtein_fully_disjoint_is_one() -> None:
    # Same length, no characters in common -> every position substituted -> 1.0.
    assert sim.normalized_levenshtein("RRRR", "UUUU") == pytest.approx(1.0)


def test_normalized_levenshtein_against_empty_is_one() -> None:
    assert sim.normalized_levenshtein("RULD", "") == pytest.approx(1.0)


# --- scanpath_similarity ---------------------------------------------------


def test_scanpath_similarity_self_is_one() -> None:
    saccs = [_sacc(0.0, 10.0), _sacc(90.0, 10.0), _sacc(180.0, 10.0)]
    assert sim.scanpath_similarity(saccs, saccs) == pytest.approx(1.0)


def test_scanpath_similarity_two_empty_is_one() -> None:
    assert sim.scanpath_similarity([], []) == pytest.approx(1.0)


def test_scanpath_similarity_disjoint_directions_is_low() -> None:
    # All-right vs. all-up encodings share no characters -> similarity 0.0.
    a = [_sacc(0.0, 10.0), _sacc(0.0, 10.0), _sacc(0.0, 10.0)]
    b = [_sacc(90.0, 10.0), _sacc(90.0, 10.0), _sacc(90.0, 10.0)]
    assert sim.scanpath_similarity(a, b) == pytest.approx(0.0)


def test_scanpath_similarity_one_differing_saccade() -> None:
    # 3 saccades, one direction flipped -> distance 1 / length 3 -> 1 - 1/3.
    a = [_sacc(0.0, 10.0), _sacc(90.0, 10.0), _sacc(180.0, 10.0)]
    b = [_sacc(0.0, 10.0), _sacc(90.0, 10.0), _sacc(-90.0, 10.0)]
    assert sim.scanpath_similarity(a, b) == pytest.approx(1.0 - 1.0 / 3.0)


def test_scanpath_similarity_long_threshold_casing_matters() -> None:
    # Same direction, amplitudes straddling the threshold -> 'R' vs 'r' differ.
    long = [_sacc(0.0, 10.0)]
    short = [_sacc(0.0, 1.0)]
    assert encode_directions(long) == "R"
    assert encode_directions(short) == "r"
    assert sim.scanpath_similarity(long, short) == pytest.approx(0.0)


def test_scanpath_similarity_on_detected_saccades_self_match() -> None:
    from itrace import saccades

    stream = _multi_saccade_stream()
    _f, saccs = saccades.detect_ivt(stream)
    assert len(saccs) > 0
    assert sim.scanpath_similarity(saccs, saccs) == pytest.approx(1.0)


# --- ngram_cosine ----------------------------------------------------------


def test_ngram_cosine_identical_is_one() -> None:
    assert sim.ngram_cosine("RULDRULD", "RULDRULD") == pytest.approx(1.0)


def test_ngram_cosine_disjoint_bigrams_is_zero() -> None:
    # No shared bigrams -> orthogonal vectors -> cosine 0.0.
    assert sim.ngram_cosine("RRRR", "UUUU") == pytest.approx(0.0)


def test_ngram_cosine_too_short_for_n_is_zero() -> None:
    # A single character has no bigrams -> zero vector -> 0.0.
    assert sim.ngram_cosine("R", "RU", n=2) == 0.0
    assert sim.ngram_cosine("", "", n=2) == 0.0


def test_ngram_cosine_partial_overlap_is_between() -> None:
    # 'RU' shares the 'RU' bigram with 'RUU' but the vectors are not identical.
    value = sim.ngram_cosine("RUU", "RRU", n=2)
    assert 0.0 < value < 1.0


def test_ngram_cosine_known_value() -> None:
    # 'RUR' bigrams: {RU:1, UR:1}; 'RUU' bigrams: {RU:1, UU:1}.
    # Shared 'RU' -> dot = 1; each norm = sqrt(2) -> cosine = 1/2.
    assert sim.ngram_cosine("RUR", "RUU", n=2) == pytest.approx(0.5)


def test_ngram_cosine_unigrams() -> None:
    # n=1 over identical multisets of characters -> 1.0.
    assert sim.ngram_cosine("RRUU", "URUR", n=1) == pytest.approx(1.0)


# --- transition_matrix -----------------------------------------------------


def test_transition_matrix_rows_sum_to_one() -> None:
    saccs = [
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
        _sacc(180.0, 10.0),  # L
    ]
    symbols, matrix = sim.transition_matrix(saccs)
    assert symbols == ["L", "R", "U"]
    assert matrix.shape == (3, 3)
    row_totals = matrix.sum(axis=1)
    # 'L' only appears last -> no outgoing transitions -> row sums to 0.
    li = symbols.index("L")
    for i in range(len(symbols)):
        expected = 0.0 if i == li else 1.0
        assert row_totals[i] == pytest.approx(expected)


def test_transition_matrix_deterministic_alternation() -> None:
    # R always -> U and U always -> R: each occupied row is a point mass.
    saccs = [
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
    ]
    symbols, matrix = sim.transition_matrix(saccs)
    assert symbols == ["R", "U"]
    ri, ui = symbols.index("R"), symbols.index("U")
    assert matrix[ri, ui] == pytest.approx(1.0)
    assert matrix[ri, ri] == pytest.approx(0.0)
    # Last symbol is 'U' so its outgoing transitions come only from index 1->2.
    assert matrix[ui, ri] == pytest.approx(1.0)


def test_transition_matrix_branching_probabilities() -> None:
    # From R: next is U once and R once -> each 0.5.
    saccs = [
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
        _sacc(0.0, 10.0),  # R
        _sacc(0.0, 10.0),  # R
        _sacc(90.0, 10.0),  # U
    ]
    symbols, matrix = sim.transition_matrix(saccs)
    ri, ui = symbols.index("R"), symbols.index("U")
    # R transitions observed: R->U, R->R, R->U  => U twice, R once over 3.
    assert matrix[ri, ui] == pytest.approx(2.0 / 3.0)
    assert matrix[ri, ri] == pytest.approx(1.0 / 3.0)
    assert matrix[ri].sum() == pytest.approx(1.0)


def test_transition_matrix_empty_saccades() -> None:
    symbols, matrix = sim.transition_matrix([])
    assert symbols == []
    assert matrix.shape == (0, 0)


def test_transition_matrix_single_saccade_no_transitions() -> None:
    symbols, matrix = sim.transition_matrix([_sacc(0.0, 10.0)])
    assert symbols == ["R"]
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == pytest.approx(0.0)


def test_transition_matrix_dtype_is_float64() -> None:
    _symbols, matrix = sim.transition_matrix([_sacc(0.0, 10.0), _sacc(90.0, 10.0)])
    assert matrix.dtype == np.float64
