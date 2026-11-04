import restore_seg


def test_slice_blocks_respects_limit():
    text = " ".join(["word"] * 700)
    blocks = restore_seg.slice_blocks(text)
    assert blocks
    assert all(len(b.split()) <= 230 for b in blocks)
    assert sum(len(b.split()) for b in blocks) == 700


def test_units_uses_the_restorer(monkeypatch):
    class FakePM:
        def restore_punctuation(self, text):
            return text.replace("great The", "great. The") + "."

    monkeypatch.setattr(restore_seg, "_PM", FakePM())
    units = restore_seg.units("The pool was great. The restaurant was awful.")
    assert len(units) == 2
    assert units[0].startswith("The pool")
