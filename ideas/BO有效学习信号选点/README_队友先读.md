# BO 有效学习信号选点

这个包检验一个问题：低 reward 提示是否一定值得训练？在 PCPO 中，一条提示的 8 张图如果全部得分相同，组内优势为零，不能提供相对好坏的更新信号。因此本包让 BO 预测“这条提示能产生多大的有效优势”，再选择尚未训练的提示。

## 与 v2 的区别

v2 的目标是提示的平均 reward；本包的目标是 PCPO 实际使用的优势信号幅度。对 8 张图的 reward，先按 PCPO 的整批标准差计算优势，再裁剪到 `[-5, 5]`，标签为 8 张图的平均绝对优势。PCPO 接收的原始 reward 不变。

## 接入位置

在 PCPO 完成一轮 64 张图的 OCR 评分后，把 8×8 reward 交给 `selector.observe()`。下一轮在 DataLoader 设置索引前调用 `selector.next_batch()`。提示特征可以直接复用 v2 的 `features.py`，不需要重新设计特征和核。

```python
from bo_signal import EffectiveSignalSelector
selector = EffectiveSignalSelector(features, seed=42)
ids = selector.next_batch()
# 原 PCPO 生图、算 reward、算优势和更新保持不变
selector.observe(ids, image_rewards)
```

预热 20 轮，之后 GP 用最近 50 轮观测，UCB 最大化有效信号。四组对照应包括 random、低平均 reward、直接按 reward 方差排序和本包的 BO 版本。这个包只验证信号选择，不声称 BO 本身一定优于直接方差排序。
