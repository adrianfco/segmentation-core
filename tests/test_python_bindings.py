"""Python binding tests for segmentation_core."""

import os
import pathlib
import tempfile

import pytest

import segmentation_core

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FOUR_COLORS = str(FIXTURES / "test_4colors.png")
GRAY = str(FIXTURES / "test_gray.png")


# ----------------------------------------------------------------
# Module import
# ----------------------------------------------------------------

def test_module_importable():
    assert hasattr(segmentation_core, "segment_image")


def test_result_type_exists():
    assert hasattr(segmentation_core, "SegmentationResult")


# ----------------------------------------------------------------
# Basic API correctness
# ----------------------------------------------------------------

def test_kmeans_returns_result(tmp_path):
    out = str(tmp_path / "out.png")
    result = segmentation_core.segment_image(
        image_path=FOUR_COLORS,
        output_path=out,
        algorithm="kmeans",
        k=4,
        seed=42,
        max_iters=100,
    )
    assert result.success
    assert result.algorithm == "kmeans"
    assert result.width  == 16
    assert result.height == 16
    assert result.runtime_ms >= 0.0
    assert result.iterations >= 1
    assert result.error_message == ""
    assert os.path.exists(out)


def test_pfcm_returns_result(tmp_path):
    out = str(tmp_path / "out.png")
    result = segmentation_core.segment_image(
        image_path=FOUR_COLORS,
        output_path=out,
        algorithm="pfcm",
        k=4,
        seed=42,
        max_iters=30,
    )
    assert result.success
    assert result.algorithm == "pfcm"
    assert os.path.exists(out)


def test_result_repr():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.png")
        r = segmentation_core.segment_image(
            image_path=FOUR_COLORS, output_path=out, k=2, seed=0, max_iters=10
        )
        rep = repr(r)
        assert "SegmentationResult" in rep


# ----------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------

def test_kmeans_reproducible(tmp_path):
    out_a = str(tmp_path / "a.png")
    out_b = str(tmp_path / "b.png")
    args = dict(image_path=FOUR_COLORS, algorithm="kmeans", k=4, seed=42, max_iters=100)
    segmentation_core.segment_image(output_path=out_a, **args)
    segmentation_core.segment_image(output_path=out_b, **args)
    assert open(out_a, "rb").read() == open(out_b, "rb").read()


def test_pfcm_reproducible(tmp_path):
    out_a = str(tmp_path / "a.png")
    out_b = str(tmp_path / "b.png")
    args = dict(image_path=FOUR_COLORS, algorithm="pfcm", k=4, seed=7, max_iters=20)
    segmentation_core.segment_image(output_path=out_a, **args)
    segmentation_core.segment_image(output_path=out_b, **args)
    assert open(out_a, "rb").read() == open(out_b, "rb").read()


# ----------------------------------------------------------------
# Error handling — result.success == False, no exception
# ----------------------------------------------------------------

def test_nonexistent_input_file(tmp_path):
    out = str(tmp_path / "out.png")
    result = segmentation_core.segment_image(
        image_path="/nonexistent/image.png",
        output_path=out,
    )
    assert not result.success
    assert result.error_message != ""


def test_invalid_k(tmp_path):
    out = str(tmp_path / "out.png")
    result = segmentation_core.segment_image(
        image_path=FOUR_COLORS,
        output_path=out,
        k=0,
    )
    assert not result.success


def test_invalid_algorithm(tmp_path):
    out = str(tmp_path / "out.png")
    with pytest.raises(Exception):
        segmentation_core.segment_image(
            image_path=FOUR_COLORS,
            output_path=out,
            algorithm="unsupported_algo",
        )


def test_empty_input_path(tmp_path):
    out = str(tmp_path / "out.png")
    result = segmentation_core.segment_image(image_path="", output_path=out)
    assert not result.success


def test_empty_output_path():
    result = segmentation_core.segment_image(
        image_path=FOUR_COLORS, output_path=""
    )
    assert not result.success


# ----------------------------------------------------------------
# Gray image (k > unique colors)
# ----------------------------------------------------------------

def test_gray_image_kmeans(tmp_path):
    out = str(tmp_path / "out.png")
    result = segmentation_core.segment_image(
        image_path=GRAY,
        output_path=out,
        k=3,
        seed=1,
        max_iters=50,
    )
    assert result.success
    assert result.width  == 8
    assert result.height == 8


# ----------------------------------------------------------------
# Output file is created
# ----------------------------------------------------------------

def test_output_file_created(tmp_path):
    out = str(tmp_path / "result.png")
    assert not os.path.exists(out)
    result = segmentation_core.segment_image(image_path=FOUR_COLORS, output_path=out)
    assert result.success
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
