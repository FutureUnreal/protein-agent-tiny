# AI4S 蛋白构象系综赛题 Baseline 上手指南

本文面向第一次接触本赛题的参赛者，目标是从零开始跑通一个可提交的 `output.zip` baseline，并理解这个 baseline 做了什么、没有做什么。

> 图片占位：这里可以放一张比赛任务流程图，展示“输入序列 -> 生成 conformers -> 打包 output.zip -> 提交评测”。

## 1. 赛题在做什么

本赛题是“蛋白质构象系综生成智能体”。初赛输入非常少：每道题只给一条单链氨基酸序列和一个参考构象数。参赛系统需要为每个蛋白生成最多 10 个三维构象，覆盖可能的动态构象状态。

官方提交物是一个 `output.zip`，根目录至少包含：

```text
1_conf1_pred.cif
1_conf2_pred.cif
...
2_conf1_pred.cif
...
3_conf1_pred.cif
...
agent.log
```

关键格式要求：

- 构象文件必须是 `.cif`，不是 `.pdb`。
- 文件命名必须是 `{problem_id}_conf{N}_pred.cif`。
- `problem_id` 是 `1`、`2`、`3`。
- 每道题最多提交 10 个 conformer。
- 每个 CIF 只能包含单条蛋白链。
- 坐标必须有限，不能有 NaN 或 Inf。
- 至少要有 CA 原子，评测会基于 CA-RMSD 和物理合理性做打分。
- `agent.log` 用于审计智能体研发流程，缺失日志可能影响成绩。

重要数据约束：

- 允许使用题目给定的氨基酸序列。
- 禁止把本次赛题的原始 MD 轨迹、晶体结构或 NMR ensemble 作为输入。
- 可以使用公开可用的预训练模型、公共力场、RCSB PDB 无关条目、AlphaFold DB、UniProt/UniRef/MGnify、无关公开 MD benchmark 等资源。

## 2. 这个 baseline 是什么

本仓库提供一个 tiny baseline：`protein-agent-tiny`。

它不是一个通用科研平台，也没有前端。它只面向本赛题，核心目标是快速生成合法的 `output.zip`，并保留一个可继续迭代的 agent loop。

baseline 默认做法：

- 读取 `data/problems/1.json`、`2.json`、`3.json`。
- 从氨基酸序列生成一组有限、单链、CA 间距合理的参数化三维 backbone 构象。
- 写出标准 mmCIF 文件。
- 写出 `agent.log`。
- 打包 `outputs/latest/output.zip`。
- 运行本地格式校验。
- 额外生成 `outputs/latest/technical_report.md` 作为技术摘要。

这个 baseline 的优点是稳定、快速、可提交；缺点是它不是高精度物理模拟，也没有默认训练深度模型。后续可以用 agent loop 或你自己的模型继续替换 `solver.py`。

> 图片占位：这里可以放一张仓库结构截图，标出 `data/problems`、`solver.py`、`scripts/run_baseline.sh`、`outputs/latest/output.zip`。

## 3. 环境要求

推荐环境：

- Linux 服务器或 WSL。
- Python 3.10。
- 能访问 GitHub 和 PyPI。
- 已安装 `git`。
- 可选 GPU。baseline 不依赖 GPU，agent 后续实验可以探测并使用 torch/CUDA。

本项目使用 `uv` 管理 Python 环境。首次运行脚本会自动安装 `uv`；如果你已经安装了，也可以直接使用。

## 4. 从 git clone 开始

```bash
git clone https://github.com/FutureUnreal/protein-agent-tiny.git
cd protein-agent-tiny
```

检查仓库内容：

```bash
ls
```

你应该能看到：

```text
data/
protein_agent_tiny/
scripts/
pyproject.toml
uv.lock
README.md
```

## 5. 一键初始化并跑 baseline

最简单方式：

```bash
bash scripts/bootstrap_server.sh
```

这个脚本会做几件事：

- 安装或检查 `uv`。
- 根据 `uv.lock` 同步 `.venv`。
- 如果没有 `.env`，从 `.env.example` 复制一份。
- 创建 `outputs/`、`workspaces/`、`runs/`、`memory/`。
- 跑一次 baseline。
- 校验 `outputs/latest/submission`。
- 打印 `output.zip` 路径。

成功时你会看到类似：

```text
PASS
agent.log: OK
output.zip: .../protein-agent-tiny/outputs/latest/output.zip
```

> 图片占位：这里可以放一张终端成功输出截图，重点圈出 `PASS` 和 `output.zip` 路径。

## 6. 只跑 baseline

如果环境已经部署好，只想重新生成提交包：

```bash
bash scripts/run_baseline.sh 1
```

参数 `1` 是 solver 的优化轮数。baseline 很快，通常几秒内完成。

输出文件：

```text
outputs/latest/output.zip
```

同时会生成：

```text
outputs/latest/submission/
outputs/latest/run_report.json
outputs/latest/technical_report.md
outputs/archive/
```

`outputs/archive/` 会保留历史快照，避免下一次运行覆盖掉之前可用的提交包。

## 7. 校验提交包

运行：

```bash
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
```

期望输出：

```text
PASS
OK 1_conf1_pred.cif
...
agent.log: OK
```

也可以检查 zip 内容：

```bash
unzip -l outputs/latest/output.zip
```

应该只看到 `.cif` 文件和 `agent.log`：

```text
1_conf1_pred.cif
1_conf2_pred.cif
...
3_conf3_pred.cif
agent.log
```

## 8. 提交哪个文件

提交这个文件：

```text
outputs/latest/output.zip
```

不要提交整个仓库、`outputs/latest/submission/` 文件夹、`run_report.json` 或 `technical_report.md`，除非比赛平台另有额外上传入口。

当前代码会生成 `technical_report.md`，用于你自己查看方法摘要；官方 zip 示例只列出 CIF 和 `agent.log`。如果平台要求额外上传技术报告，可以单独上传：

```text
outputs/latest/technical_report.md
```

## 9. 配置 LLM Agent

baseline 不需要 LLM。只有你要运行 agent 自动迭代时，才需要配置 `.env`。

编辑 `.env`：

```bash
cp .env.example .env
nano .env
```

填写 OpenAI-compatible 接口：

```text
OPENAI_API_KEY=你的密钥
OPENAI_API_BASE=你的 OpenAI-compatible base URL
PROTEIN_AGENT_MODEL=你的模型名
OPENALEX_API_KEY=
```

`OPENALEX_API_KEY` 是可选的，用于更稳定地检索文献。

不要把 `.env` 提交到 Git。仓库已经默认忽略 `.env`。

## 10. 运行 Agent 迭代

Agent 会在 workspace 中复制一份 `solver.py`，然后执行“研究计划 -> 假设 -> 改代码 -> 跑实验 -> 反思”的循环。

示例：

```bash
bash scripts/run_agent.sh 2 20 1
```

参数含义：

```text
scripts/run_agent.sh <agent_iterations> <max_minutes_per_iteration> <solver_candidate_rounds>
```

例如：

```bash
bash scripts/run_agent.sh 20 45 1
```

表示：

- 最多 20 轮 agent 迭代。
- 每轮最多 45 分钟。
- 每次 solver 运行 1 轮候选优化。

Agent 运行结束后，仍然输出：

```text
outputs/latest/output.zip
```

中间过程在：

```text
workspaces/<timestamp>/
runs/
memory/
```

这些目录是本地运行产物，不应提交到 Git。

> 图片占位：这里可以放一张 agent workspace 截图，展示 `research_plan.md`、`hypothesis.md`、`iteration_result_*.json`、`solver_diff_*.patch`。

## 11. GPU 和 torch

仓库的 `pyproject.toml` 默认包含 torch：

```text
torch>=2.2.0,<2.7
```

如果服务器有 NVIDIA GPU，环境探测会写入：

```text
workspaces/<timestamp>/environment_report.md
```

你可以查看：

```bash
cat workspaces/<timestamp>/environment_report.md
```

需要注意：

- baseline 不需要 GPU。
- torch 可用不代表 agent 一定会自动使用 GPU。
- 如果 agent 认为当前最稳的路径是参数化 CPU solver，它可能仍然不使用 GPU。
- 真正的 GPU 模型训练需要你或 agent 在 `solver.py` 中实现对应逻辑，并保持无 GPU 时的 fallback。

## 12. 记录官方分数

提交到平台后，可以把官方分数记录进 memory，供下一次 agent 读取。

手动输入：

```bash
bash scripts/record_score.sh --score 0.85 --score1 0.87 --score2 0.88 --success true --notes "first baseline"
```

或粘贴官方 JSON：

```bash
bash scripts/record_score.sh --score-json '{"score":0.85,"scoreJson":{"score1":0.87,"score2":0.88},"success":true,"errorMsg":""}'
```

这会写入：

```text
memory/scores.jsonl
```

`memory/` 是本地状态，不提交 Git。

## 13. 常见问题

### Q1: 为什么没有生成 10 个 conformer？

每题最多 10 个，不要求必须 10 个。题目 JSON 里有 `conformer_count` 参考值，baseline 默认按参考数量生成。

### Q2: `technical_report.md` 在哪里？

```text
outputs/latest/technical_report.md
```

也可以手动重新生成：

```bash
.venv/bin/python -m protein_agent_tiny.report --run-dir outputs/latest
```

### Q3: `agent.log` 里会不会有密钥？

不应该有。代码会对提交日志做路径和序列脱敏，也不会把 `.env` 写入 `agent.log`。你提交前仍可以自己检查：

```bash
unzip -p outputs/latest/output.zip agent.log | grep -E 'OPENAI|API_KEY|sk-|你的密钥片段'
```

正常情况下没有输出。

### Q4: baseline 算作弊吗？

baseline 只使用题目给定序列，不使用本次赛题的 MD、晶体结构或 NMR ensemble。它会写 `agent.log` 说明数据政策和运行过程。是否满足最终平台审计要求，以比赛官方解释为准。

### Q5: Windows 可以跑吗？

建议 Linux/WSL。脚本是 Bash，Windows 原生 PowerShell 下需要改命令或使用 WSL。

## 14. 最小命令清单

只想跑通 baseline：

```bash
git clone https://github.com/FutureUnreal/protein-agent-tiny.git
cd protein-agent-tiny
bash scripts/bootstrap_server.sh
bash scripts/run_baseline.sh 1
.venv/bin/python -m protein_agent_tiny.validate --submission-dir outputs/latest/submission
ls -lh outputs/latest/output.zip
```

要跑 agent：

```bash
cp .env.example .env
# 编辑 .env，填 OPENAI_API_KEY、OPENAI_API_BASE、PROTEIN_AGENT_MODEL
bash scripts/run_agent.sh 20 45 1
ls -lh outputs/latest/output.zip
```

提交：

```text
outputs/latest/output.zip
```

