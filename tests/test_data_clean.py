from __future__ import annotations

from app.services.chunking.splitter import split_items
from app.services.data_clean import clean_items, clean_table_text, clean_text


def test_clean_text_removes_invisible_chars_emoji_and_mojibake() -> None:
    assert clean_text("飞书\u200b一键养🦞虾🎉 Â â€™") == "飞书一键养虾"


def test_clean_table_text_cleans_json_cells_without_breaking_json() -> None:
    raw = (
        '[{"功能":"★ 一键部署🔥",'
        '"说明":"欢迎大家加入飞书养虾群交流分享🎉 '
        '图片：data\\\\processing\\\\OpenClaw 真一键部署来了！\\\\img\\\\image_0002.png '
        '同步自文档: https://example.feishu.cn/docx/source"}]'
    )

    assert clean_table_text(raw) == '[{"功能":"一键部署","说明":"欢迎大家加入飞书养虾群交流分享"}]'


def test_clean_items_removes_source_sync_link_and_duplicate_image_path_line() -> None:
    items = [
        {"type": "paragraph", "style": "标题", "text": "OpenClaw 真一键部署来了！"},
        {
            "type": "paragraph",
            "style": "正文",
            "text": "欢迎大家加入飞书养虾群交流分享🎉",
        },
        {
            "type": "paragraph",
            "style": "正文",
            "text": "图片：data\\processing\\OpenClaw 真一键部署来了！\\img\\image_0003.jpg",
        },
        {
            "type": "image",
            "style": "图片",
            "text": "data\\processing\\OpenClaw 真一键部署来了！\\img\\image_0003.jpg",
            "path": "data\\processing\\OpenClaw 真一键部署来了！\\img\\image_0003.jpg",
            "description": "宣传海报对比本地部署与飞书妙搭方案。",
        },
        {"type": "paragraph", "style": "正文", "text": "同步自文档: {{同步自文档网站链接}}"},
        {
            "type": "link_ref",
            "style": "链接",
            "text": "https://example.feishu.cn/docx/source（同步自文档网站链接）",
            "url": "https://example.feishu.cn/docx/source",
            "description": "同步自文档网站链接",
        },
    ]

    cleaned = clean_items(items)

    assert [item["type"] for item in cleaned] == ["paragraph", "paragraph", "image"]
    assert cleaned[1]["text"] == "欢迎大家加入飞书养虾群交流分享"


def test_split_items_applies_cleaning_before_chunking() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题", "text": "2.3 🔥 来飞书妙搭，一键养虾吧！"},
            {"type": "paragraph", "style": "正文", "text": "立即开启养「虾」之旅吧！🎁"},
            {"type": "paragraph", "style": "正文", "text": "同步自文档: {{同步自文档网站链接}}"},
            {
                "type": "link_ref",
                "style": "链接",
                "text": "https://example.feishu.cn/docx/source",
                "url": "https://example.feishu.cn/docx/source",
                "description": "同步自文档网站链接",
            },
        ]
    )

    assert len(chunks) == 1
    assert "🔥" not in chunks[0].metadata["path"]
    assert "🎁" not in chunks[0].content
    assert "同步自文档" not in chunks[0].content
    assert "links" not in chunks[0].metadata
