# encoding=utf-8
"""
n-gram 语言模型：
用“最近的 n-1 个 token（词/字）”来预测下一个 token 的概率。

比如 trigram（n=3）模型，预测当前词，只看前面两个词。
"""
import random
from collections import defaultdict


def simple_n_gram():
    """
    简化版 伪n-gram，超迷你版“按前缀续写句子”
    有点像最原始的“马尔可夫链文本生成”，不过是用字符级来做的：
    我们关心的东西是：“给定前缀 text，下一字符是什么”
    这个代码的做法就是在语料里统计：
    在历史上出现过多少次 “textX” 这样的片段（X 为一个字符）
    然后把所有这些 X 拿出来，从中随机抽一个作为当前的“下一字符”

    这就是一个 非常简化的“基于前缀的马尔可夫生成”：
    状态：当前前缀 text
    可能的下一状态：text + 某个字符
    转移概率：由语料中出现的频次隐性决定（因为随机从样本里采）
    """
    corpus = [
        "我喜欢学习", "我会跑步", "我想吃苹果",
        "你喜欢做题", "你想吃草莓", "你会打羽毛球",
        "他喜欢读书", "他会游泳", "他想吃香蕉",
        "她喜欢上课", "她会打乒乓球", "她想喝牛奶",
        "我们喜欢背单词", "我们想吃葡萄", "我们会踢足球",
        "大家喜欢写字", "大家会打篮球", "大家想吃西瓜",
        "哥哥会爬山", "哥哥想喝可乐",
        "姐姐会骑车", "姐姐想喝茶",
        "朋友喜欢听讲", "朋友会滑冰", "朋友想喝咖啡",
        "同学会滑雪", "同学想喝果汁",
        "老师喜欢考试",
        "学生喜欢提问",
        "孩子喜欢进步"]

    prefix = "我"
    n = 3

    text = prefix
    for _ in range(n):
        # 所有候选“下一个字符”的列表
        opts = []
        for s in corpus:
            # 在句子 s 中滑动窗口，找 “text” 出现的位置
            for i in range(len(s) - len(text)):
                if s[i:i + len(text)] == text:
                    opts.append(s[i + len(text)])
        if not opts: break
        print(f"opts = {opts}")
        text += random.choice(opts)

    print(text)


def common_n_gram():
    """
    字符级 trigram 模型（n=3）
    训练时收集所有长度为 3 的 n-gram
    推理时基于概率分布采样下一个字符

    相比前面的简单版本，该标准版本可以做到：
    ✔固定窗口（n-1） → 不会越来越难匹配
    ✔ 构建统计表 → 有“学习”
    ✔ 按概率采样 → 更符合语言分布
    ✔ 可以泛化 → 可以生成未见过的组合
    ✔ 有句首/句尾符号 → 生成更自然
    ✔ 可以扩展到按词生成（效果提升巨大）
    """
    corpus = [
        "我喜欢学习", "我会跑步", "我想吃苹果",
        "你喜欢做题", "你想吃草莓", "你会打羽毛球",
        "他喜欢读书", "他会游泳", "他想吃香蕉",
        "她喜欢上课", "她会打乒乓球", "她想喝牛奶",
    ]

    N = 3  # n-gram 的 n
    prefix = "我喜"  # 初始前缀（必须 >= n-1）
    length = 10  # 最终想生成的字符数

    # ---------------------
    # 1. 训练：构建 n-gram 概率表
    # ---------------------
    counts = defaultdict(lambda: defaultdict(int))

    for s in corpus:
        s = "^" * (N - 1) + s + "$"  # 加句首/句尾符号，提高鲁棒性
        for i in range(len(s) - N + 1):
            history = s[i:i + N - 1]  # 前 n-1 个字符（状态）
            next_char = s[i + N - 1]  # 下一个字符（输出）
            counts[history][next_char] += 1

    # ---------------------
    # 2. 基于概率采样下一个字符
    # ---------------------
    def sample_next(history):
        if history not in counts:
            return None
        next_chars = list(counts[history].keys())
        freqs = list(counts[history].values())
        # 按频率加权随机选择
        return random.choices(next_chars, weights=freqs)[0]

    # ---------------------
    # 3. 生成文本
    # ---------------------
    text = prefix
    while len(text) < length:
        history = text[-(N - 1):]  # 用最新的 n-1 个字符作为条件
        next_char = sample_next(history)
        if not next_char or next_char == "$":
            break
        text += next_char

    print("生成结果：", text)

if __name__ == '__main__':
    common_n_gram()