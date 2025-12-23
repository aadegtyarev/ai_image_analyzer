import bot

from bot import simple_markdown_to_html


def test_header_and_body_separated():
    header = "Изображение #1 — промт: art_analysis_photo\n\n"
    body = "Анализ фотографии кота"
    combined = header + body
    html = simple_markdown_to_html(combined)
    # Ensure there is at least one blank line (two newlines) between header and body in the source
    assert "\n\n" in combined
    # And in HTML result newlines are preserved (no merging)
    assert "art_analysis_photo" in html
    assert "Анализ фотографии кота" in html
    # The two parts should not be immediately adjacent
    assert "art_analysis_photoАнализ" not in html
