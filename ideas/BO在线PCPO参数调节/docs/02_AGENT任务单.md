# 给接入 agent 的任务单

1. 阅读输入契约，确认上下文五个字段顺序。这个包不使用提示特征，直接读取训练状态。
2. 把 `suggest()` 接到每 10 或 20 轮的控制点。
3. 把学习率写入现有 optimizer param groups。
4. 把 beta 写入 PCPO 的 KL loss 项。
5. 用留出验证集计算 `validation_delta` 和实际 KL，调用 `observe()`。
6. 保存与恢复同 epoch 的 tuner 状态，先跑 21 轮再跑 250 轮。

只有第 4、5 步依赖队友 PCPO 脚本；BO 模型、候选网格、状态和学习率 hook 已实现。
