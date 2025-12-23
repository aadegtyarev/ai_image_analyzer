from bot import parse_tail_flags


def test_parse_tail_flags_simple():
    clean, g = parse_tail_flags("group")
    assert g is True and clean == ""


def test_parse_tail_flags_with_text():
    clean, g = parse_tail_flags("group please analyze")
    assert g is True and clean == "please analyze"


def test_parse_tail_flags_no_flag():
    clean, g = parse_tail_flags("some text")
    assert g is False and clean == "some text"
