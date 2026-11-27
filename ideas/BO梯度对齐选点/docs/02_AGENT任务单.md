# 给接入 agent 的任务单

1. 先读两个特征文档，再确认 Dataset 行号和 HDF5 顺序；特征文件使用 `integration/features.py` 和 `integration/assets/feature_schema.json`。
2. 将 `next_batch()` 接到 sampler。
3. 从原 PCPO loss 提取每个提示组的 LoRA 梯度；不要重新定义 reward 或优势。
4. 用固定训练参考集计算 `reference_gradient`，不能读取最终测试集。
5. 使用 `flatten_parameter_grads()` 和固定随机投影，调用 `selector.observe()`。
6. 保存与 PCPO epoch 对齐的状态，跑 21 轮后再跑 250 轮。

只有第 3、4 步需要改队友的 PCPO 训练脚本，其余组件已实现并有测试。
