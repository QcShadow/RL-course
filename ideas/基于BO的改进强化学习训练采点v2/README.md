# 实验包入口

先读 `README_队友先读.md`，再按 `docs/01_接入步骤.md`、`docs/02_接口与方法说明.md` 操作。CPU 验证命令是：

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python -B -m unittest discover -s tests -v
PYTHONPATH=. python -B examples/cpu_smoke.py --acquisition ucb
```

这个包包含可直接替换到现有 PCPO 单 GPU 入口的接入文件；正式训练前先按文档跑 21 轮试跑，再跑完整预算。
