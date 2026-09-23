import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q1.alignment import (
    align_text_words,
    convolution_spans,
    make_time_grid,
    overlap_pool,
    standardize_aligned,
)


def test_time_grid_is_half_open_and_monotonic():
    grid = make_time_grid(10.0, 5)
    assert grid.tolist() == [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [8.0, 10.0]]


def test_overlap_pool_uses_time_not_array_index():
    values = np.asarray([[1.0], [3.0]], dtype=np.float32)
    spans = np.asarray([[0.0, 1.0], [3.0, 4.0]], dtype=np.float64)
    grid = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    pooled, mask, _ = overlap_pool(values, spans, grid)
    assert np.allclose(pooled[:, 0], [1.0, 3.0])
    assert mask.tolist() == [1, 1]


def test_unmatched_words_are_kept_with_fallback_timestamps():
    records = align_text_words(
        "hello missing world",
        [{"text": "hello", "start": 0.0, "end": 0.5}, {"text": "world", "start": 1.0, "end": 1.5}],
        2.0,
    )
    assert len(records) == 3
    assert records[1]["fallback"] is True
    assert records[1]["start"] >= records[0]["end"]
    assert records[1]["end"] <= records[2]["start"]


def test_wavlm_receptive_field_is_derived_from_config():
    spans, valid, timing = convolution_spans(4, 16000, [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2], 1.0)
    assert timing == {"effective_stride_samples": 320, "receptive_field_samples": 400}
    assert spans.shape == (4, 2)
    assert valid.tolist() == [True, True, True, True]


def test_standardization_excludes_masked_padding():
    values = np.asarray([[[1.0], [0.0]], [[3.0], [0.0]]], dtype=np.float32)
    masks = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    output, mean, std, _ = standardize_aligned(values, masks)
    assert np.isclose(mean[0], 2.0)
    assert np.isclose(std[0], 1.0)
    assert np.allclose(output[:, 1], 0.0)
