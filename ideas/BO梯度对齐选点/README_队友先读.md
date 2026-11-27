# BO 梯度对齐选点

接入前先读 [特征与接入对齐](docs/00_特征与接入对齐.md)。

这个包检验一个不同于 reward 选点的问题：一条提示的平均 reward 很低，不代表它产生的 PCPO 更新方向适合整个 OCR 任务。本包让 BO 预测“提示组的 PCPO 梯度是否和一组固定参考提示的梯度一致”。

## 目标定义

对选中提示组计算 PCPO 损失对 LoRA 参数的梯度，拼成向量 `g_x`。在一个固定的小型训练参考集上计算参考梯度 `g_ref`，观测为余弦相似度：

```text
alignment = dot(g_x, g_ref) / (||g_x|| * ||g_ref||)
```

BO 最大化这个观测。PCPO 的 reward、优势、截断目标和 KL 项保持原样；BO 只决定下一轮使用哪些尚未训练的提示。

## 接入位置和成本

队友需要在完成一轮 reward 计算后，保留本轮各提示组的 PCPO loss，并用 `torch.autograd.grad` 得到每组 LoRA 梯度；再在固定参考集上得到一次 `g_ref`，调用 `selector.observe(ids, group_gradients, reference_gradient)`。不能用最终测试集作为参考集。

完整梯度很大，建议只收集 LoRA 参数并做固定随机投影到 64 或 128 维后再存储。这个方法增加反向计算和显存，不增加生图；先用 1 个 seed、21 轮试跑验证接口，再决定是否跑完整预算。

## 已接好和待接入

已接好：梯度展平、固定随机投影、有限值检查、余弦对齐、BO 选点、状态保存和恢复。

需要队友在 PCPO 脚本补三处：保留每个提示组的 PCPO loss、对 LoRA 参数调用 `torch.autograd.grad`、在参考集上计算 `reference_gradient`。这三处依赖训练脚本里的 loss、参数和参考数据对象，包内留空；不能用最终测试集代替参考集。

## 对照

至少比较 random、v2 的 reward 选点、本包的直接梯度对齐排序，以及本包的 BO 预测。这样才能分清提升来自梯度指标还是来自 BO。
