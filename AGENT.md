# 项目目标与工作契约

## 研究目标

参考 `docs/paper_plan/jev_ship_ffd_paper_plan.md`，将 Jev-like 决策模型 Kev 引入船舶设计中的自然语言驱动 FFD 任务。

系统将自然语言转换为船型 FFD 动作，覆盖：

- 模糊语义离散化；
- 多区域操作；
- 多步、连续编辑；
- 工程约束判断；
- FFD/NURBS 动作执行与几何检查。

Kev 作为船型 FFD 的类型化决策层，负责在候选答案中进行判断。

## Kev 架构

保留 Kev 当前的 Jev-like pointer-head 决策架构：

```text
状态 + 类型化问题
    -> 候选答案概率
    -> 最大概率答案
    -> FFD 动作投影
```

训练数据使用 Kev labelled request 格式：`state + questions + label`。
统一动作投影包含：

```json
{
  "actions": [
    {"region": "bow", "operation": "outward"}
  ]
}
```

## 动作语义

- 区域：`bow`、`stern`、`bulb`、`midbody`、`deck`、`bilge`、`global`；
- 操作：方向、丰满度、外飘、球鼻艏长度、全船长度和全船宽度操作；
- 模糊意图：离散 `magnitude_level`；
- 明确意图：`explicit` 模式及精确数值字段。

当前训练重点是类型化判断。精确 `magnitude_value`、`longitudinal_extent`、`vertical_extent` 的直接预测列为后续研究事项。

## 评估协议

1. **所有模型统一**：使用同一份 `test_5k.jsonl`、同一 Kev 问题提示词、同一概率 JSON 格式、同一概率校验和同一 `kev.benchmark` 汇总代码。
2. **原生 JEV**：输出类型化问题的概率，报告题目级回答准确率、概率/校准指标，并将最大概率答案投影为 FFD 动作后报告 FFD 准确率。
3. **本地 Kev 与语言模型**：通过模型各自的本地或远程 provider 经 pi 调用，输出同一类型化问题概率，报告同一组题目级概率/校准指标；必要时统一将最大概率答案投影为 FFD 动作并报告 FFD 准确率。

共同 FFD 主指标包含：

- 动作数量；
- 按顺序的 `region`；
- 按顺序的 `operation`。

幅度、作用范围和对称性属于扩展动作字段，当前不纳入共同主指标。概率指标单独报告。

### 分母与失败记录

- turn-level 分母使用请求记录数；question-level 分母使用展开后的 Kev 题目数；
- 格式错误、缺字段、调用失败、非法概率和动作数量错误均计为失败记录；
- 失败记录进入 FFD 指标分母；
- 真实动作数量为 1、预测数量为 2/3 时，记录为预测失败；
- 题目级 `clean.acc` 与统一 FFD 主指标分开报告。

## 数据集与运行记录

- 结构化动作由程序生成，自然语言由语言数据生成流程扩展；
- 支持单动作、多区域单轮和多步编辑；
- 训练、验证、测试按船型集合隔离；
- 数据 manifest、SHA-256、代码版本、运行 revision 和 Job ID 随训练/评估记录；
- 删除或覆盖结果前检查运行状态、目标冲突并列出清单；
- 代码、测试、配置和文档提交 Git，数据集、模型权重、环境和生成结果不提交。

## 文件存放

- 脚本：`scripts/`；Slurm 任务脚本只放 `scripts/slurm/`
- 代码：`src/`，按功能分子目录
- 模型客户端：`src/model_clients/`
- 可视化代码：`visualization/ffd/`、`visualization/reconstruction/`
- 数据集、模型权重、运行输出不入库

## 开发原则

- 先检查代码、数据格式、训练入口和报告字段，再修改；
- 采用最小必要改动；
- 训练协议、评估协议和统一评分保持一致；
- 改变 Kev 架构、训练目标、评分分母或数据语义前先确认；
- 未经明确同意不执行 `git push`、`git rebase`、`git reset --hard`。

## 当前状态

- Kev 类型化决策训练与统一概率评估为主线；
- 原生 JEV、Kev、本地模型和远程语言模型使用同一 Kev 概率问答评估协议；
- 统一概率结果可进一步投影为共同 FFD 动作指标；
- 精确数值直接输出暂缓。
