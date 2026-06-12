from __future__ import annotations

from app.services.parser.powerpoint_parser import ParseContext, _extract_picture


class _Shape:
    def __init__(self, image) -> None:
        self.image = image


class _UnsupportedImage:
    @property
    def content_type(self) -> str:
        raise ValueError("unsupported image format, expected one of: ..., got 'MPO'")

    @property
    def blob(self) -> bytes:
        raise AssertionError("unsupported images should be skipped before reading blob")


class _UnknownContentTypeImage:
    content_type = "image/mpo"

    @property
    def blob(self) -> bytes:
        raise AssertionError("unknown image content types should be skipped before reading blob")


def test_powerpoint_parser_skips_images_when_content_type_raises(tmp_path) -> None:
    context = ParseContext(source_path=tmp_path / "demo.pptx", image_dir=tmp_path)

    item = _extract_picture(
        _Shape(_UnsupportedImage()),
        context,
        slide_index=1,
        shape_order=2,
        slide_width=100,
        slide_height=100,
    )

    assert item is None
    assert context.image_index == 0
    assert list(tmp_path.iterdir()) == []


def test_powerpoint_parser_skips_unknown_image_content_types(tmp_path) -> None:
    context = ParseContext(source_path=tmp_path / "demo.pptx", image_dir=tmp_path)

    item = _extract_picture(
        _Shape(_UnknownContentTypeImage()),
        context,
        slide_index=1,
        shape_order=2,
        slide_width=100,
        slide_height=100,
    )

    assert item is None
    assert context.image_index == 0
    assert list(tmp_path.iterdir()) == []
