# test_jieba_cut.py
import jieba

texts = [
    "一书一课",
    "一书一课管理后台",
    "一书一课管理后台的登录网址是什么？",
    "一书一课管理后台操作流程.docx",
]

print("=== 默认 jieba 分词 ===")
for text in texts:
    print(f"\n原文：{text}")
    print("精确模式：", list(jieba.cut(text, cut_all=False)))
    print("搜索引擎模式：", list(jieba.cut_for_search(text)))
    print("全模式：", list(jieba.cut(text, cut_all=True)))

print("\n=== 添加自定义词后 ===")
custom_words = [
    "一书一课",
    "一书一课管理后台",
    "管理后台",
]

for word in custom_words:
    jieba.add_word(word, freq=100000, tag="nz")

for text in texts:
    print(f"\n原文：{text}")
    print("精确模式：", list(jieba.cut(text, cut_all=False)))
    print("搜索引擎模式：", list(jieba.cut_for_search(text)))
    print("全模式：", list(jieba.cut(text, cut_all=True)))