# 给接入 agent 的任务单

1. 阅读固定特征和变长分组说明；特征文件使用 `integration/features.py` 和 `integration/assets/feature_schema.json`。
2. 将 `next_batch()` 返回的 `prompt_indices` 和 `image_counts` 接到 sampler。
3. 用 `expand_prompt_indices()` 展开 Dataset 行号，保持总数 64。
4. 用 `group_flat_rewards()` 归组 reward，不能 `reshape(8, 8)`。
5. 修改 PCPO 优势统计器，使每组使用自己的图片数量。
6. 检查 batch size、梯度累积和 optimizer 更新次数，跑 21 轮后再跑 250 轮。

第 2、5 步依赖队友 PCPO 训练脚本；其余接口和归组代码已实现并有测试。
