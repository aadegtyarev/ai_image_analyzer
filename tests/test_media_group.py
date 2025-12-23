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


def test_media_context_update_on_override(monkeypatch, tmp_path):
    """If a prompt override or prompt file is provided after the first image, cached context must update."""
    import bot as B
    # initial context
    B.set_media_context('mgX', {'system_prompt': 'S', 'prompt_label': 'art', 'use_text_override': False, 'user_text': None})

    # override with text
    B.update_media_context_with_override('mgX', None, 'override text', prompt_debug=True)
    ctx = B.get_media_context('mgX')
    assert ctx is not None
    assert ctx.get('use_text_override') is True
    assert ctx.get('user_text') == 'override text'
    assert ctx.get('prompt_label') == 'текст из сообщения'

    # override with prompt file
    p = tmp_path / 'special.txt'
    p.write_text('SPECIAL PROMPT\n', encoding='utf-8')
    B.update_media_context_with_override('mgX', str(p), None, prompt_debug=True)
    ctx2 = B.get_media_context('mgX')
    assert ctx2 is not None
    assert ctx2.get('prompt_label') == 'special'
    assert 'SPECIAL PROMPT' in (ctx2.get('system_prompt') or '')
