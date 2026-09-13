from polymimus.cli import _fmt_time, _to_srt, _to_txt, _to_vtt
from polymimus.transcribe import Segment, TranscriptionResult


def make_result() -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        language_probability=0.98,
        segments=[
            Segment(start=0.0, end=2.5, text="Hello world."),
            Segment(start=3.0, end=3661.75, text="Goodbye."),
        ],
        duration=3662.0,
    )


def test_fmt_time_zero():
    assert _fmt_time(0.0) == "00:00:00.000"


def test_fmt_time_full():
    assert _fmt_time(3661.75) == "01:01:01.750"


def test_fmt_time_subsecond():
    assert _fmt_time(0.5) == "00:00:00.500"


def test_to_txt():
    assert _to_txt(make_result()) == "Hello world.\nGoodbye."


def test_to_srt():
    srt = _to_srt(make_result())
    lines = srt.split("\n")
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "Hello world."
    assert lines[3] == ""
    assert lines[4] == "2"
    assert lines[5] == "00:00:03,000 --> 01:01:01,750"


def test_to_vtt():
    vtt = _to_vtt(make_result())
    lines = vtt.split("\n")
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""
    assert lines[2] == "00:00:00.000 --> 00:00:02.500"
    assert lines[3] == "Hello world."
