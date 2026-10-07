# NL2Hull 中文说明

NL2Hull 面向自然语言驱动的约束船型设计，将船体的 NURBS 表示、自由变形（Free-Form Deformation，FFD）和类型化决策连接起来。

## 项目概览

项目将船舶设计语言转换为可执行的几何操作，主要包含：

- 船体水线和艏艉纵剖面的归一化 NURBS 表示；
- 面向船体区域、操作类型、变形等级和保持约束的类型化决策；
- FFD 动作投影与连续控制点变形；
- 船体重建、几何合法性检查和约束评估；
- 统一的决策准确率、概率校准、动作精确匹配和端到端执行评测。

英文主说明见 [README.md](README.md)。

## 方法流程

系统将一条设计请求表示为有序的类型化决策序列。Kev/Jev 风格的决策接口选择动作数量及其属性，FFD 引擎随后在归一化 NURBS 船体上执行这些操作。

```text
自然语言请求
      ↓
区域 / 操作 / 幅度 / 约束决策
      ↓
有序 FFD 动作
      ↓
NURBS 控制点变形
      ↓
船体重建与约束检查
```

几何表示采用三次 NURBS 水线、艏艉纵剖面、保持对称性的剖面构造和局部平滑影响窗口。评测过程将调用失败、非法概率、格式错误动作和几何失败纳入相应分母。

## 数据集

项目使用十二种归一化经典船型，并维护以下数据目录：

- `datasets/classic_hulls/`：原始与标准化船体几何、元数据和基准清单；
- `datasets/ship_design_structured_actions/`：30,000 条程序生成的结构化 FFD 动作；
- `datasets/ship_design_decisions/`：134,558 条清洗后的语言—决策记录，包含训练集、验证集、测试集和固定的 `test_5k` 子集；
- `datasets/jevbench/`：评测协议使用的固定 JevBench 公共文件；
- `datasets/kev_decision_v7/`：Kev decision-v7 上游数据集，用于原始预训练阶段。

船舶设计决策数据集包含 88,604 条训练记录、22,758条验证记录和23,196条测试记录。数据按船型家族划分：训练集使用 DTC、DTMB 5415、KCS、KVLCC2、集装箱船、护卫舰和 NPL 变体；验证集使用 S-175 与 Series 60；测试集使用 Wigley 与 workboat 船型。

`datasets/ship_design_decisions/` 已按后续发布 Hugging Face 数据集的需要整理，清单中记录文件哈希、划分规模、来源记录和清洗过程。

## 快速开始

安装开发版本：

```bash
python -m pip install -e .
```

运行经典船型重建基准：

```bash
python -m nurbs_ship_reconstruction.cli \
  --dataset datasets/classic_hulls \
  --output outputs/local_reconstruction
```

运行 Wigley 船型快速测试：

```bash
python -m nurbs_ship_reconstruction.cli \
  --dataset datasets/classic_hulls \
  --hull wigley_hull \
  --levels 12 \
  --output outputs/wigley_smoke
```

运行测试：

```bash
python -m pytest
```

## 目录结构

```text
src/nurbs_ship_reconstruction/  NURBS 表示、船体重建和 FFD 引擎
src/dataset/                    结构化数据和语言数据处理
src/benchmarks/                 统一动作与概率评测
src/end2end/                    动作执行与端到端评测
src/model_clients/              决策服务和本地模型适配器
scripts/evaluate/               评测入口
scripts/slurm/                  集群训练与评测任务
tests/                          单元测试和集成测试
visualization/                  论文图片与诊断图生成
datasets/                       本地数据与清单
outputs/                        本地实验输出
```

## 论文与复现

论文工作目录为 `docs/paper/`，图片集中在 `docs/paper/pics/`，因此 LaTeX 源文件在制作 arXiv 压缩包和后续投稿材料时使用统一的本地图片路径。当前保留一份主论文源文件；当投稿期刊要求不同模板时，再增加对应的模板封装或元数据文件。

实验输出中的清单记录数据哈希、代码版本、模型信息和评测状态。结合固定数据划分与这些清单，可以复现论文中的实验结果。

## 引用

arXiv 记录和仓库元数据确定后补充正式引用信息。
