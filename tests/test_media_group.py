from bot import set_media_context, get_media_context
import time


def test_media_context_basic():
    set_media_context('mg1', {'system_prompt': 'S', 'prompt_label': 'art', 'use_text_override': False, 'user_text': None})
    ctx = get_media_context('mg1')
    assert ctx is not None
    assert ctx['prompt_label'] == 'art'


def test_media_context_ttl():
    set_media_context('mg2', {'system_prompt': 'S2', 'prompt_label': 'portrait', 'use_text_override': False, 'user_text': None})
    # artificially expire
    import bot as B
    B.MEDIA_CONTEXTS['mg2']['ts'] -= (B.MEDIA_CONTEXT_TTL + 1)
    assert get_media_context('mg2') is None


def test_media_context_logging(monkeypatch, capsys):
    monkeypatch.setenv("DEBUG", "1")
    set_media_context('mg3', {'system_prompt': 'S3', 'prompt_label': 'series', 'use_text_override': False, 'user_text': None})
    captured = capsys.readouterr()
    assert "[MEDIA_DEBUG] set media context mg3" in captured.err
    # expire and check cleanup log
    import bot as B
    B.MEDIA_CONTEXTS['mg3']['ts'] -= (B.MEDIA_CONTEXT_TTL + 1)
    B._cleanup_media_contexts()
    captured = capsys.readouterr()
    assert "[MEDIA_DEBUG] expired media context mg3" in captured.err


def test_set_media_context_without_user_text(monkeypatch):
    """Ensure caching prompt info when no user_text is present doesn't raise."""
    monkeypatch.setenv("DEBUG", "1")
    # simulate handler branch where prompt_path is used and no user_text is set
    import bot as B
    try:
        # should not raise
        B.set_media_context('mg4', {'system_prompt': 'S4', 'prompt_label': 'art', 'use_text_override': False})
    except Exception as e:
        raise AssertionError(f"set_media_context raised unexpectedly: {e}")
