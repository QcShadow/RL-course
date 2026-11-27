# 实验包入口

先读 `README_队友先读.md`，再按 `docs/00_特征与接入对齐.md`、`docs/01_部署与接入.md` 和 `docs/02_AGENT任务单.md` 操作。CPU 验证命令是：

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -B -m unittest discover -s tests -v
PYTHONPATH=. python -B examples/cpu_smoke.py
```

选点和梯度处理模块已经完成；队友需要在 PCPO 脚本保留组内 loss，并按任务单先跑 21 轮单 GPU 试跑。
