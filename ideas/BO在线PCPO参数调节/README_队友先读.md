# BO 在线 PCPO 参数调节

接入前先读 [输入与接入对齐](docs/00_特征与接入对齐.md)。

这个包把 BO 放到 PCPO 的训练控制层：每隔若干轮从一个小的安全候选网格里选择 KL 系数和学习率。PCPO 的优势计算、稳定目标、时间步 credit 和 KL 项都保留；BO 只决定下一阶段使用哪一组参数。

## 目标和输入

`context` 是当前训练状态，例如验证集 OCR reward、最近训练 reward、KL、clipfrac 和优势方差。一次试验结束后交回：

```text
validation_delta = 下一次独立验证 reward - 当前验证 reward
kl = 这段训练的实际 KL
```

BO 在 `context + log(KL系数) + log(learning_rate)` 上拟合 GP。启用 `kl_budget` 时，会同时拟合 KL 预测并惩罚超过预算的候选。

## 接入步骤

队友在训练入口初始化 `PCPOParameterBO`，每 10 或 20 轮调用 `suggest(context)`。把返回的 `kl_coefficient` 写入 PCPO 的 KL 项，把 `learning_rate` 写入优化器；经过一个固定窗口后，在留出验证集上计算 `validation_delta`，再调用 `observe()`。模型检查点、BO 状态和验证窗口必须使用同一个轮次。

不要用最终测试集作为反馈，不要每轮尝试很多参数，也不要把参数改变后的训练 reward 当作泛化效果。这个方向有在线 BO 调 RL 超参数的先例，实验重点是它是否适合 PCPO-OCR，而不是声称 BO 调参本身新颖。

## 已接好和待接入

已接好：固定上下文字段读取、候选参数网格、UCB 和 KL 预算惩罚、状态保存和恢复、优化器学习率写入。

需要队友在 PCPO 脚本补两处：把 `selected_kl_coefficient(choice)` 写入当前 loss 的 KL 系数；在固定验证窗口结束后计算 `validation_delta` 和实际 KL，再调用 `observe()`。这两处依赖训练脚本中的 loss 和评测对象，包内不伪造实现。

## 推荐对照

固定原配置、手工网格中最优固定参数、无约束 BO 和带 KL 预算的 BO。所有方法使用相同 seed、总训练图片数和验证预算。
