# 小红书宣传文案：ICML 2026 Spotlight 版

## 标题

ICML 2026 Spotlight：FlashTrace 给长推理一张证据地图

## 封面文案

```text
ICML 2026 Spotlight
FlashTrace

给长推理一张证据地图
130x faster
```

副标题：

```text
看清模型回答时依赖了哪些输入
```

## 正文

ICML 2026 Spotlight：FlashTrace

Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs

现在的 reasoning LLM 会写很长的思考、推导和代码。答案看起来很完整，研究者更想知道：它到底参考了哪些输入？

长推理链让可解释性变难。最终答案往往先依赖中间 reasoning，再间接依赖原始 prompt 和上下文。

FlashTrace 做的是长链路溯源：从最终答案出发，先找到关键推理步骤，再回到原始输入证据。

一个关键设计是 span-wise aggregation：把一整段输出一起解释，让多 token attribution 变成一次更高效的计算。

另一个关键设计是 recursive attribution：沿着答案 -> 推理链 -> 输入，一跳一跳回溯信息来源。

论文实验里，5k token target span 下 FlashTrace 在 20 秒内完成，IFR 超过 38 分钟，速度提升超过 130x。

我们在长上下文检索、数学推理、多跳问答和 Aider 代码生成上做了验证，希望让 reasoning agent 的行为更容易被检查、调试和审计。

工具包支持 Python API、CLI、JSON trace 和 HTML heatmap：

Paper: https://arxiv.org/abs/2602.01914

GitHub: https://github.com/wbopan/flashtrace

PyPI:

```bash
pip install flashtrace
```

#LLM #Interpretability #可解释性 #ReasoningLLM #ChainOfThought #FlashTrace #ICML2026 #Spotlight #AI论文 #TokenAttribution
