# encoding=utf-8

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

"""
TF-IDF（Term Frequency – Inverse Document Frequency）是一种衡量“一个词对一篇文档的重要程度”的统计指标
TF（词频）衡量一个词在文档中出现得多不多，词t在文档d中出现的次数/文档d的总词数
IDF（逆文档频率）衡量一个词在所有文档中有多“稀有”，log(N/(1+nₜ))，N=所有文档的数量，nₜ=包含词t的文档数量，log 是为了缩放到合理范围。

两者相乘：
| 情况        | TF | IDF | TF-IDF 结果 | 说明               |
| ---------- | -- | ---- | ---------- | ----------------- |
| 词频高且稀有 | 高  | 高   | 最高       | 文档主题词          |
| 词频高但常见 | 高  | 低   | 中等       | “的”“我们”类词被抑制 |
| 词频低但稀有 | 低  | 高   | 中等       | 提到一次的专有名词    |
| 词频低且常见 | 低  | 低   | 最低       | 完全不重要，信息量最低 |

TF-IDF 现在常用于：
- 作为baseline（基线模型）来对比深度模型；
- 作为大模型前的数据预筛选（比如先用 TF-IDF 快速选候选，再用 BERT 精排）；
- 作为可解释分析工具（给人看权重分布、主题词）。

TF-IDF 是一个“老而弥坚”的算法。它仍然在：
- 搜索引擎底层；
- 文本聚类与检索；
- 小规模 NLP 系统；
- 可解释性分析中发挥作用。

但它不理解语义，只是统计。在需要理解上下文、情感、逻辑的任务里，它已被深度语言模型取代。
"""

if __name__ == '__main__':
    docs = [
        "苹果手机维修服务，提供屏幕更换、电池更换、主板维修与进水处理。",
        "三星手机维修服务，支持屏幕维修、电池更换与主板维修。",
        "北京今日天气晴朗，气温适宜，蓝天白云，非常适合出行。"
    ]

    top_k = 10  # 设置取几个关键词
    tok = lambda s: [w for w in jieba.lcut(s) if len(w) > 1 and w.strip()]

    v = TfidfVectorizer(tokenizer=tok, token_pattern=None)
    X = v.fit_transform(docs)  # 如果有多篇文档，就用docs列表替换[doc]即可
    vocab = v.get_feature_names_out()

    # 提取TF（scikit-learn的TfidfVectorizer默认是词频/文档长度）
    tf = (X > 0).astype(int).toarray()[0] * 0  # 先占位
    term_counts = np.array(X.toarray()[0]) / v.idf_  # TF（因为TF-IDF / IDF = TF）
    tf = term_counts / term_counts.sum()  # 归一化后就是TF

    idf = v.idf_  # 每个词的IDF
    tfidf = X.toarray()[0]

    # 取权重最高的top_k
    idx = tfidf.argsort()[::-1][:top_k]

    for i in idx:
        print(f"{vocab[i]}\tTF: {tf[i]:.3f}\tIDF: {idf[i]:.3f}\tTF-IDF: {tfidf[i]:.3f}")
