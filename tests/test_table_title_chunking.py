import re

from app.services.chunking.splitter import split_items
from app.models.chunk import Chunk
from app.services.embedding import EmbeddingService, _format_content_for_embedding
from app.services.parser.word_parser import format_extracted_items


def test_format_extracted_items_writes_table_title_before_table_json() -> None:
    text = format_extracted_items(
        [
            {
                "type": "image_table",
                "style": "图片表格",
                "table_name": "年度销售表",
                "text": '[{"月份":"1月","销售额":"100"}]',
            }
        ]
    )

    assert text.splitlines() == [
        "[table] [表标题] 年度销售表",
        '[image_table] [图片表格] [{"月份":"1月","销售额":"100"}]',
    ]


def test_table_title_line_is_added_to_next_table_chunk_path() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题 1", "text": "经营数据"},
            {"type": "table", "style": "表标题", "text": "年度销售表"},
            {"type": "image_table", "style": "图片表格", "text": '[{"月份":"1月","销售额":"100"}]'},
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "table"
    assert chunks[0].metadata["path"] == "经营数据\\年度销售表"
    assert chunks[0].content == '[{"月份":"1月","销售额":"100"}]'


def test_short_text_chunks_are_folded_and_merged_under_parent_path() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题 1", "text": "行业发展"},
            {"type": "paragraph", "style": "标题 2", "text": "第一阶段"},
            {"type": "paragraph", "style": "正文", "text": "萌芽期以经验积累为主。"},
            {"type": "paragraph", "style": "标题 2", "text": "第二阶段"},
            {"type": "paragraph", "style": "正文", "text": "工业化推动技术快速迭代。"},
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_type"] == "text"
    assert chunks[0].metadata["path"] == "行业发展"
    assert "第一阶段\n萌芽期以经验积累为主。" in chunks[0].content
    assert "第二阶段\n工业化推动技术快速迭代。" in chunks[0].content


def test_consecutive_same_level_heading_fragments_are_not_overwritten() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题", "text": "中华人民共和国"},
            {"type": "paragraph", "style": "标题", "text": "国家标准"},
            {"type": "paragraph", "style": "标题", "text": "钢材断口检验法"},
            {
                "type": "paragraph",
                "style": "正文",
                "text": "本标准规定了钢材断口检验的试样制备和检验方法。",
            },
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["path"] == (
        "中华人民共和国 国家标准 钢材断口检验法"
    )
    assert chunks[0].content == (
        "本标准规定了钢材断口检验的试样制备和检验方法。"
    )


def test_nonconsecutive_same_level_heading_still_replaces_previous_section() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题", "text": "第一章"},
            {"type": "paragraph", "style": "正文", "text": "第一章正文内容足够形成切块。"},
            {"type": "paragraph", "style": "标题", "text": "第二章"},
            {"type": "paragraph", "style": "正文", "text": "第二章正文内容足够形成切块。"},
        ]
    )

    paths = [chunk.metadata["path"] for chunk in chunks]
    assert paths == ["第一章", "第二章"]


def test_json_chunk_path_is_rewritten_when_its_heading_is_folded() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题 1", "text": "行业发展"},
            {"type": "paragraph", "style": "标题 2", "text": "第一阶段"},
            {"type": "paragraph", "style": "正文", "text": "萌芽期以经验积累为主。"},
            {"type": "table", "style": "表标题", "text": "阶段数据"},
            {"type": "image_table", "style": "图片表格", "text": '[{"阶段":"萌芽期"}]'},
        ]
    )

    text_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "text"]
    table_chunks = [chunk for chunk in chunks if chunk.metadata["chunk_type"] == "table"]
    assert len(text_chunks) == 1
    assert len(table_chunks) == 1
    assert text_chunks[0].metadata["path"] == "行业发展"
    assert table_chunks[0].content == '[{"阶段":"萌芽期"}]'
    assert table_chunks[0].metadata["path"] == "行业发展\\阶段数据"


def test_image_metadata_stays_with_image_description_after_split() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题 1", "text": "操作说明"},
            {"type": "paragraph", "style": "正文", "text": "前文" * 180},
            {
                "type": "image",
                "style": "图片",
                "text": "data/processing/demo/img/image_0001.png",
                "path": "data/processing/demo/img/image_0001.png",
                "description": "点击保存按钮完成提交。",
            },
            {"type": "paragraph", "style": "正文", "text": "后文" * 180},
        ]
    )

    image_chunks = [chunk for chunk in chunks if "<img " in chunk.content]
    non_image_chunks = [chunk for chunk in chunks if "<img " not in chunk.content]
    assert len(image_chunks) == 1
    assert re.fullmatch(r'<img index="\d+">图片：点击保存按钮完成提交。</img>', image_chunks[0].content)
    assert image_chunks[0].metadata["imgs"][0]["img_path"] == "data/processing/demo/img/image_0001.png"
    assert all("imgs" not in chunk.metadata for chunk in non_image_chunks)


def test_long_image_description_keeps_tags_and_image_metadata_on_each_piece() -> None:
    long_description = "这是很长的图片描述。" * 80
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题 1", "text": "操作说明"},
            {
                "type": "image",
                "style": "图片",
                "text": "data/processing/demo/img/image_0002.png",
                "path": "data/processing/demo/img/image_0002.png",
                "description": long_description,
            },
        ]
    )

    assert len(chunks) > 1
    assert all(re.match(r'<img index="\d+">', chunk.content) for chunk in chunks)
    assert all(chunk.content.endswith("</img>") for chunk in chunks)
    assert all(chunk.metadata["imgs"][0]["img_path"] == "data/processing/demo/img/image_0002.png" for chunk in chunks)


def test_img_tags_are_removed_from_embedding_text() -> None:
    content = '<img index="3">图片：点击保存按钮完成提交。</img>'

    assert _format_content_for_embedding(content) == "点击保存按钮完成提交。"


def test_link_markers_become_indexed_tags_with_embedding_link_text() -> None:
    chunks = split_items(
        [
            {"type": "paragraph", "style": "标题 1", "text": "工具链接"},
            {"type": "paragraph", "style": "正文", "text": "秀米：{{秀米网站链接}}"},
            {
                "type": "link_ref",
                "style": "链接",
                "text": "https://xiumi.us/#/",
                "url": "https://xiumi.us/#/",
                "description": "秀米网站链接",
            },
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].content == '秀米：\n<a index="2">链接：秀米网站链接</a>'
    assert chunks[0].metadata["links"] == [
        {
            "index": 2,
            "link_name": "秀米网站链接",
            "link_path": "https://xiumi.us/#/",
        }
    ]
    assert _format_content_for_embedding(chunks[0].content) == "秀米：\n秀米网站链接"


def test_embedding_text_uses_file_name_stem_path_and_content_only() -> None:
    chunk = Chunk(
        content="正文内容",
        metadata={
            "file_name": "产品手册.docx",
            "file_path": "data/raw/产品手册.docx",
            "path": "章节\\小节",
            "file_id": "file_123",
            "links": [{"link_name": "链接名", "link_path": "https://example.com"}],
        },
    )

    assert EmbeddingService().build_embedding_text(chunk) == "产品手册 章节\\小节 正文内容"


def test_table_embedding_text_uses_json_array_rows() -> None:
    content = '[{"制造商":"Pfaudler","总部地点":"美国"},{"制造商":"De Dietrich Process Systems","总部地点":"法国"}]'
    chunk = Chunk(
        content=content,
        metadata={"file_name": "供应商.xlsx", "path": "供应商\\玻璃衬里反应釜"},
    )

    assert EmbeddingService().build_embedding_text(chunk) == (
        '供应商 供应商\\玻璃衬里反应釜 '
        '[["制造商","总部地点"],["Pfaudler","美国"],["De Dietrich Process Systems","法国"]]'
    )


def test_table_embedding_drops_repeated_header_row_from_model_json() -> None:
    content = (
        '[{"序号":"序号","牌号":"牌号"},'
        '{"序号":"1","牌号":"S30210"},'
        '{"序号":"2","牌号":"S30403"}]'
    )
    chunk = Chunk(
        content=content,
        metadata={"file_name": "标准.pdf", "path": "表 2"},
    )

    assert EmbeddingService().build_embedding_text(chunk) == (
        '标准 表 2 [["序号","牌号"],["1","S30210"],["2","S30403"]]'
    )
