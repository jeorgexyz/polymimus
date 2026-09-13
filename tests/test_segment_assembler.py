from polymimus.mic import SegmentAssembler

SPEECH = b"\x01\x00" * 480
SILENCE = b"\x00\x00" * 480


def feed_frames(assembler, frames):
    """Feed (frame, is_speech) pairs; return all completed segments."""
    segments = []
    for frame, is_speech in frames:
        out = assembler.feed(frame, is_speech)
        if out is not None:
            segments.append(out)
    return segments


def test_silence_only_never_emits():
    assembler = SegmentAssembler(silence_frames=10)
    assert feed_frames(assembler, [(SILENCE, False)] * 100) == []


def test_speech_then_silence_emits_one_segment():
    assembler = SegmentAssembler(silence_frames=10)
    frames = [(SPEECH, True)] * 20 + [(SILENCE, False)] * 20
    segments = feed_frames(assembler, frames)
    assert len(segments) == 1


def test_segment_includes_pre_speech_padding():
    # a few silence frames precede speech; the ring buffer should keep them
    assembler = SegmentAssembler(silence_frames=10)
    frames = [(SILENCE, False)] * 3 + [(SPEECH, True)] * 20 + [(SILENCE, False)] * 20
    segments = feed_frames(assembler, frames)
    assert len(segments) == 1
    # padding + speech + trailing silence before cutoff => longer than speech alone
    assert len(segments[0]) > 20 * len(SPEECH)


def test_two_utterances_emit_two_segments():
    assembler = SegmentAssembler(silence_frames=5)
    utterance = [(SPEECH, True)] * 15 + [(SILENCE, False)] * 15
    segments = feed_frames(assembler, utterance * 2)
    assert len(segments) == 2


def test_brief_noise_does_not_trigger():
    # a single noisy frame in silence should not open a segment
    assembler = SegmentAssembler(silence_frames=10)
    frames = [(SILENCE, False)] * 10 + [(SPEECH, True)] * 2 + [(SILENCE, False)] * 30
    assert feed_frames(assembler, frames) == []


def test_min_ring_size_is_one():
    # silence_frames=0 must not crash (clamped to 1)
    assembler = SegmentAssembler(silence_frames=0)
    frames = [(SPEECH, True)] * 5 + [(SILENCE, False)] * 2
    segments = feed_frames(assembler, frames)
    assert len(segments) == 1
