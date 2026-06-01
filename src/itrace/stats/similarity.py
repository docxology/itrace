"""Scanpath comparison and string-similarity metrics.

This module quantifies how alike two recorded scanpaths are, operating on the
direction-encoded representation produced by :mod:`itrace.encoding`. A saccade
sequence is turned into a string of cardinal-direction characters (``R``/``L``/
``U``/``D``, upper-case for long saccades, lower-case for short ones) and the
strings are then compared with classic sequence-similarity measures:

* **Edit distance** -- :func:`levenshtein` (raw) and
  :func:`normalized_levenshtein` (scaled to ``[0, 1]``).
* **Scanpath similarity** -- :func:`scanpath_similarity`, ``1 - normalized
  edit distance`` directly over two saccade lists (``1.0`` = identical).
* **N-gram structure** -- :func:`ngram_cosine`, the cosine similarity of two
  n-gram count vectors, capturing local-pattern overlap regardless of global
  alignment.
* **Transition structure** -- :func:`transition_matrix`, the first-order
  symbol-transition probability matrix of a single scanpath.

Everything is pure NumPy: no matplotlib, no optional dependency, deterministic.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..encoding import encode_directions, ngram_counts
from ..types import FloatArray, Saccade


def levenshtein(a: str, b: str) -> int:
    """Levenshtein (edit) distance between two strings.

    The minimum number of single-character insertions, deletions or
    substitutions needed to turn ``a`` into ``b``, computed with the classic
    iterative dynamic-programming recurrence in ``O(len(a) * len(b))`` time and
    ``O(len(b))`` space.

    Parameters
    ----------
    a, b:
        The strings to compare. Either may be empty.

    Returns
    -------
    int
        The edit distance; ``0`` iff ``a == b``.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution / match
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein(a: str, b: str) -> float:
    """Levenshtein distance scaled to ``[0, 1]`` by the longer string length.

    Parameters
    ----------
    a, b:
        The strings to compare. Either may be empty.

    Returns
    -------
    float
        ``levenshtein(a, b) / max(len(a), len(b))``; ``0.0`` when both strings
        are empty (no difference). ``0.0`` means identical, ``1.0`` means
        maximally dissimilar.
    """
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return levenshtein(a, b) / longest


def scanpath_similarity(
    saccades_a: list[Saccade],
    saccades_b: list[Saccade],
    *,
    long_threshold_deg: float = 5.0,
) -> float:
    """Direction-string similarity between two scanpaths.

    Both saccade lists are encoded to direction strings via
    :func:`itrace.encoding.encode_directions`; the result is ``1 -
    normalized_levenshtein`` of those strings, so identical scanpaths score
    ``1.0`` and completely dissimilar ones approach ``0.0``.

    Parameters
    ----------
    saccades_a, saccades_b:
        The two saccade sequences to compare. Either may be empty.
    long_threshold_deg:
        Amplitude threshold passed to the encoder (long vs. short casing).

    Returns
    -------
    float
        Similarity in ``[0, 1]``; ``1.0`` for identical encodings (including two
        empty scanpaths).
    """
    seq_a = encode_directions(saccades_a, long_threshold_deg=long_threshold_deg)
    seq_b = encode_directions(saccades_b, long_threshold_deg=long_threshold_deg)
    return 1.0 - normalized_levenshtein(seq_a, seq_b)


def ngram_cosine(seq_a: str, seq_b: str, n: int = 2) -> float:
    """Cosine similarity of two encoded scanpaths' n-gram count vectors.

    Each string is reduced to a bag of contiguous n-grams via
    :func:`itrace.encoding.ngram_counts`; the cosine of the two count vectors
    (over the union of observed n-grams) measures local-pattern overlap
    independent of global alignment.

    Parameters
    ----------
    seq_a, seq_b:
        Encoded direction strings (e.g. from
        :func:`itrace.encoding.encode_directions`).
    n:
        N-gram length (must be >= 1).

    Returns
    -------
    float
        Cosine similarity in ``[0, 1]``; ``0.0`` when either string is shorter
        than ``n`` (no n-grams) and thus has a zero vector.

    Raises
    ------
    ValueError
        If ``n < 1`` (propagated from :func:`itrace.encoding.ngram_counts`).
    """
    counts_a = ngram_counts(seq_a, n)
    counts_b = ngram_counts(seq_b, n)
    if not counts_a or not counts_b:
        return 0.0
    vocabulary = sorted(set(counts_a) | set(counts_b))
    vec_a = np.array([counts_a.get(g, 0) for g in vocabulary], dtype=np.float64)
    vec_b = np.array([counts_b.get(g, 0) for g in vocabulary], dtype=np.float64)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def transition_matrix(
    saccades_list: list[Saccade],
    long_threshold_deg: float = 5.0,
) -> tuple[list[str], FloatArray]:
    """First-order direction-transition probability matrix of a scanpath.

    The saccade list is encoded to a direction string; the empirical
    first-order transition counts over its characters are row-normalised so each
    row is a probability distribution over the next symbol. The symbol alphabet
    is the sorted set of characters that actually occur in the encoding.

    Parameters
    ----------
    saccades_list:
        Saccades to encode and analyse. May be empty.
    long_threshold_deg:
        Amplitude threshold passed to the encoder (long vs. short casing).

    Returns
    -------
    tuple of (list of str, FloatArray)
        ``(symbols, matrix)`` where ``symbols`` is the sorted alphabet and
        ``matrix[i, j]`` is ``p(symbol_j | symbol_i)``. Each row sums to ``1.0``
        when its source symbol was observed as a transition origin, else to
        ``0.0`` (a symbol that only ever appears last has no outgoing
        transitions). For fewer than two saccades the alphabet still reflects
        the encoding and the matrix is all zeros.
    """
    seq = encode_directions(saccades_list, long_threshold_deg=long_threshold_deg)
    symbols = sorted(set(seq))
    k = len(symbols)
    matrix = np.zeros((k, k), dtype=np.float64)
    if k == 0:
        return symbols, matrix
    index = {ch: i for i, ch in enumerate(symbols)}
    for current, nxt in pairwise(seq):
        matrix[index[current], index[nxt]] += 1.0
    row_totals = matrix.sum(axis=1)
    nonzero = row_totals > 0.0
    matrix[nonzero] /= row_totals[nonzero, None]
    return symbols, matrix
