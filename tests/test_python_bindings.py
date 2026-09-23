"""Python binding tests for segmentation_core."""

import os
import pathlib
import random
import threading
import time

import pytest

import segmentation_core
from make_fixture import write_png

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FOUR_COLORS = str(FIXTURES / "test_4colors.png")
GRAY = str(FIXTURES / "test_gray.png")


def test_module_importable():
    assert hasattr(segmentation_core, "segment_image")


def test_result_type_exists():
    assert hasattr(segmentation_core, "SegmentationResult")


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


def test_result_repr(tmp_path):
    r = segmentation_core.segment_image(
        image_path=FOUR_COLORS, output_path=str(tmp_path / "out.png"), k=2, seed=0, max_iters=10
    )
    assert "SegmentationResult" in repr(r)


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


def test_output_file_created(tmp_path):
    out = str(tmp_path / "result.png")
    assert not os.path.exists(out)
    result = segmentation_core.segment_image(image_path=FOUR_COLORS, output_path=out)
    assert result.success
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


@pytest.fixture(scope="module")
def busy_image(tmp_path_factory):
    """Noise image large enough that one segmentation is observable from another thread.

    The checked-in fixtures are 16x16 and 8x8 and segment in microseconds.
    """
    width = height = 256
    rng = random.Random(0)
    pixels = [
        (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        for _ in range(width * height)
    ]
    path = tmp_path_factory.mktemp("busy") / "busy.png"
    write_png(str(path), width, height, pixels)
    return str(path)


def test_gil_released_during_segmentation(busy_image, tmp_path):
    counter = 0
    stop = threading.Event()

    def spin():
        nonlocal counter
        while not stop.is_set():
            counter += 1

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        time.sleep(0.05)  # let the spinner get scheduled before we measure

        # sleeping definitely releases the GIL, so this is how fast the spinner
        # runs when nothing is holding it, on whatever machine we are on
        start, t0 = counter, time.perf_counter()
        time.sleep(0.1)
        idle_rate = (counter - start) / (time.perf_counter() - t0)

        start, t0 = counter, time.perf_counter()
        result = segmentation_core.segment_image(
            image_path=busy_image,
            output_path=str(tmp_path / "out.png"),
            algorithm="kmeans",
            k=16,
            seed=1,
            max_iters=100,
        )
        busy_rate = (counter - start) / (time.perf_counter() - t0)
    finally:
        stop.set()
        spinner.join()

    assert result.success
    # if the call held the GIL the spinner would only get the one switch
    # interval either side of it, a few percent of its idle rate
    assert busy_rate > idle_rate / 4


def test_concurrent_calls_match_serial(busy_image, tmp_path):
    args = dict(image_path=busy_image, algorithm="kmeans", k=8, seed=3, max_iters=40)

    serial_out = str(tmp_path / "serial.png")
    segmentation_core.segment_image(output_path=serial_out, **args)
    expected = open(serial_out, "rb").read()

    results = []
    lock = threading.Lock()

    def run(index):
        out = str(tmp_path / f"threaded_{index}.png")
        result = segmentation_core.segment_image(output_path=out, **args)
        with lock:
            results.append((result, out))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    for result, out in results:
        assert result.success
        assert open(out, "rb").read() == expected
