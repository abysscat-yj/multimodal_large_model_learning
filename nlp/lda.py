# encoding=utf-8
"""
对10 条中文短文本自动做主题分类（Topic Modeling），
用 LDA（Latent Dirichlet Allocation）潜在狄利克雷分配模型，发现“隐藏主题”，
用关键词字典自动给主题命名，最终输出每句话的主题概率、主题名字和 Top 词。

其核心思想是将每篇文档视为多个主题的概率混合，而每个主题则由词汇的概率分布构成。
LDA通过建立文档-主题分布与主题-词分布，利用概率推断方法（如变分推断或Gibbs采样）对参数进行估计，从而自动识别文档中最有代表性的主题集合。

具体流程：
原始文本 docs
      ↓ 分词 + 清洗
词袋矩阵 X（CountVector）
      ↓
LDA 训练
      ↓ 得到两个矩阵
      - doc_topic：文档 → 主题分布
      - topic_word：主题 → 单词权重
      ↓
每个主题挑 Top 词
      ↓
用规则库给主题自动命名
      ↓
输出：每句话属于哪个主题、概率、主题名、关键字
"""
import re, jieba, numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = [
    "这家餐厅味道很好，主打川菜和火锅，环境也不错",
    "昨天去了新开的咖啡馆，拿铁顺滑，甜点也很惊喜",
    "球队本赛季状态火热，主教练强调高位逼抢与传控",
    "人工智能正在改变医疗影像分析与药物发现的流程",
    "新出的手机拍照很强，夜景模式和人像模式都清晰",
    "我在健身房练器械和有氧，主要是减脂和增肌计划",
    "电影节连续放映多部文艺片，摄影与配乐都很出色",
    "电商平台的物流很快，售后客服也专业，购物体验好",
    "研究者提出新的预训练模型框架，用于文本生成任务",
    "这家面馆的牛肉面和小菜分量足，价格也很实惠",
]

# 让 LDA 自动学习 4 个主题。LDA 是一个“主题数量必须提前指定”的模型（超参数）
# LDA 会把文档聚成 4 类主题，最终输出 4 个主题，每个主题包含词的概率
N_TOPICS = 6

TOPN_WORDS = 10

# 特征采用 1-gram（unigram），即单词级，不使用 bigram、trigram。(1,1) = 只用单词，不用词组
NGRAM_RANGE = (1, 1)

# 自动给主题命名时，至少要匹配到 2 个关键词才能“命名成功”
MIN_HITS = 2

# 对主题没有意义的词
STOPWORDS = """
的 了 在 和 与 也 都 很 比较 非常 可能 我们 他们 正在 新的 新出 新开 本赛季 这家 这个
不错 主要 计划 连续 多部 环境 模式 清晰 专业 平台 体验
""".split()
stop = set(STOPWORDS)

# 领域词典先加入
for w in ["咖啡馆", "拿铁", "甜点"]:
    jieba.add_word(w)


def tok(s: str):
    # 仅保留中文、英文和数数字，然后分词，再过滤停用词与纯数字、长度为1的token
    s = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]+", " ", s)
    seg = jieba.lcut(s)
    return [w for w in seg if len(w) > 1 and (not w.isdigit()) and (w not in stop)]

# 把文档转成 词频矩阵（Bag-of-Words）
vec = CountVectorizer(tokenizer=tok, token_pattern=None,
                      ngram_range=NGRAM_RANGE, max_df=0.95, min_df=1)
X = vec.fit_transform(docs)
terms = np.array(vec.get_feature_names_out())

lda = LatentDirichletAllocation(
    n_components=N_TOPICS, random_state=42, max_iter=200,
    doc_topic_prior=0.1,  # 越小越更偏少数主题
    topic_word_prior=0.01  # 越小越Top词更集中
)
doc_topic = lda.fit_transform(X)
comp = lda.components_


def top_terms(C, vocab, topn=10):
    out = []
    for k in range(C.shape[0]):
        idx = np.argsort(C[k])[::-1][:topn]
        out.append(vocab[idx].tolist())
    return out


topic_words = top_terms(comp, terms, topn=TOPN_WORDS)

# 用知识库给 LDA 主题贴上中文自然语义标签
LABEL_SEEDS = {
    "餐饮美食": {"火锅", "餐厅", "面馆", "牛肉面", "川菜", "味道", "实惠"},
    "咖啡甜品": {"咖啡馆", "拿铁", "甜点"},
    "体育赛事": {"球队", "主教练", "传控", "逼抢", "赛季", "状态"},
    "影视娱乐": {"电影节", "文艺片", "摄影", "配乐", "放映"},
    "电商服务": {"电商", "物流", "售后", "客服", "购物", "体验", "平台"},
    "数码影像": {"拍照", "夜景", "人像", "手机", "清晰"},
    "AI科技": {"人工智能", "预训练", "模型", "文本生成", "医疗影像", "药物发现", "框架"},
    "健身健康": {"健身房", "器械", "有氧", "减脂", "增肌"}
}


def name_topic(words, min_hits=MIN_HITS):
    s = set(words)
    best_label, best_hits = "未命名", 0
    for label, keyset in LABEL_SEEDS.items():
        hits = len(s & keyset)
        if hits > best_hits:
            best_label, best_hits = label, hits
    return best_label if best_hits >= min_hits else "未命名"


topic_names = [name_topic(ws) for ws in topic_words]

top_idx = doc_topic.argmax(1)
top_prob = doc_topic.max(1)

rows = []
for i, txt in enumerate(docs):
    k = top_idx[i]
    rows.append({
        "原句": txt,
        "主题编号": int(k),
        "主题名称": topic_names[k],
        "该主题概率": float(round(top_prob[i], 3)),
        "该主题Top词": ", ".join(topic_words[k][:8])
    })
df = pd.DataFrame(rows)

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.unicode.east_asian_width', True)

topic_overview = pd.DataFrame({
    "主题编号": list(range(N_TOPICS)),
    "主题名称": topic_names,
    "Top词（按权重降序）": [", ".join(ws) for ws in topic_words]
})
print("\n================ 主题总览 ================\n")
print(topic_overview.to_string(index=False))

print("\n================ 文档归属（逐行） ================\n")
for i, row in df.iterrows():
    print(f"[Doc{i}] 原句：{row['原句']}")
    print(f"       主题：{row['主题编号']} | {row['主题名称']} | 概率：{row['该主题概率']}")
    print(f"       Top词：{row['该主题Top词']}\n")

print("\n================ 文档归属（整表） ================\n")
print(df.to_string(index=False))
