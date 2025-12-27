import pytest

from ai_image_analyzer import call_model_with_text_only, call_model_with_image


def test_call_model_text_only_raises():
    with pytest.raises(RuntimeError):
        call_model_with_text_only({}, "hello")


def test_call_model_with_image_raises():
    with pytest.raises(RuntimeError):
        call_model_with_image({}, b"abc")
