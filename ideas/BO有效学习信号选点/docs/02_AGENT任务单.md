# 给接入 agent 的任务单

1. 阅读 `README_队友先读.md` 和 `docs/00_特征与接入对齐.md`。
2. 使用包内的 `integration/features.py` 和 `integration/assets/feature_schema.json`，让编码输入和训练 Dataset 的提示行序一致。
3. 在 PCPO sampler 迭代前调用 `next_batch()`，在 reward gather 后调用 `observe_flat()`。
4. 按 PCPO epoch 保存和恢复 selector 状态，不能读取 latest 状态。
5. 先跑 CPU 测试，再做 21 轮单 GPU 试跑；确认日志后再跑 250 轮。

PCPO 的 reward、优势、loss、优化器和测试评测不改。完成后回传启动命令、状态目录、测试日志和 250 轮结果。
