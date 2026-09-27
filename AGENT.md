# Project goals and working contract

## Primary research goal

参考 `docs/paper_plan/jev_ship_ffd_paper_plan.md`，将 Jev-like 决策模型 Kev 引入船舶设计中的自然语言驱动 FFD 任务。

系统将用户自然语言转换为船型 FFD 动作，支持：

- 不清晰语义的离散化决策；
- 多区域操作；
- 多步、连续编辑；
- 工程约束相关判断；
- 后续由 FFD/NURBS 几何程序执行动作并检查结果。

Kev 的定位是船型 FFD 的**类型化决策层**，不是直接生成大量 NURBS 控制点的回归模型。

## Kev architecture decision

保留当前 Kev 的 Jev-like 决策能力和 pointer-head 架构：

```text
状态 + 类型化问题
    -> 各问题的候选答案概率
    -> 取最大概率答案
    -> 投影为统一 FFD 动作
```

不要将 Kev 改造成标准自回归 JSON 生成模型。修改语言模型输出头会使模型从候选答案判断器变成结构化文本生成器，失去当前 Kev/Jev-like 的概率决策接口和校准能力。统一 JSON 只作为动作投影和评估层的表示，不作为 Kev 的训练输出头。

Kev 训练数据继续使用 Kev 的 labelled request 格式：`state + questions + label`。动作字段通过类型化问题表示，当前共同 FFD 主指标投影以下字段：

```json
{
  "actions": [
    {"region": "bow", "operation": "outward"}
  ]
}
```

## Action semantics

当前动作空间包含：

- 区域：`bow`、`stern`、`bulb`、`midbody`、`deck`、`bilge`、`global`；
- 操作：方向操作、丰满度/外飘/球鼻艏长度操作，以及全船长度/宽度操作；
- 定性强度：离散 `magnitude_level`；
- 显式数值模式：数据集可以标记 `explicit` 模式并包含精确数值字段。

当前 Kev 训练和评估将显式数值作为离散模式/类型化字段处理，**不能据此宣称模型具备精确数值回归和精确范围输出能力**。精确 `magnitude_value`、`longitudinal_extent`、`vertical_extent` 的直接预测列为后续事项，考虑训练成本暂不处理。

## Evaluation contract

四类对象必须区分评估协议：

1. **原生 JEV**：回答类型化问题，产生逐题概率；可报告题目级回答准确率、JEV 特有的概率/校准指标，并将最大概率答案投影为 FFD 动作后报告 FFD 准确率。
2. **本地 Kev**：使用 Kev `DecisionModel`/`LocalPredictor` 回答类型化问题；可报告题目级回答准确率、概率/校准指标和共同 FFD 准确率。
3. **提示词驱动的大语言模型**：当前协议要求直接输出动作 JSON；主指标是共同 FFD 准确率，不自动声称具备 Kev/JEV 概率指标。
4. **API 大语言模型**：若使用与提示词大语言模型相同的动作 JSON 协议，使用相同的 FFD 评分，不通过 Kev 概率题协议伪造概率指标。

共同 FFD 主指标当前只比较：

- 动作数量；
- 按顺序的 `region`；
- 按顺序的 `operation`。

不将精确幅度、作用范围或对称性混入当前共同主指标。概率指标单列，不要求动作 JSON 模型生成概率。

### Denominator and failures

- 请求记录数是 turn-level 分母；展开后的 Kev 题目数是 question-level 分母，不能混用。
- 格式错误、缺字段、调用失败、非法概率、预测动作数量超过真实动作槽位，均计为该记录失败并进入分母。
- 真实动作数量为 1 而模型预测 2/3 个动作时，判定预测失败，不访问不存在的题目字段。
- FFD 的完整 turn 正确率必须包含失败记录；成功记录上的题目级 `clean.acc` 不能替代统一 FFD 主指标。

## Dataset and split policy

- 结构化动作数据由程序生成，自然语言由语言数据生成流程扩展；
- 支持单动作、多区域单轮和多步编辑；
- 训练/验证/测试按船型集合隔离，不能仅按随机文本切分；
- 训练样本、测试样本和评估输出必须记录数据 manifest、SHA-256、代码版本和运行 revision；
- 旧结果只有在确认无用、无运行冲突并列出清单后才可删除；当前任务输出不得静默覆盖。

## Development rules

- 先检查现有代码、数据格式、训练入口和输出报告，再修改；
- 保持最小必要改动，不为未提出的需求增加兼容层或 fallback；
- 任何会改变 Kev 模型架构、训练目标、统一评分分母或数据集语义的决定，先确认后实现；
- 训练和评估代码必须保持协议一致，禁止把历史 JEV/概率评估结果与直接动作 JSON 结果混为同一指标；
- 代码类改动提交 Git，数据集、模型权重、虚拟环境和生成结果不提交；
- 未经明确同意不执行 `git push`、`git rebase`、`git reset --hard` 或覆盖远端结果。

## Current status

- Kev 的类型化决策训练和本地评估协议继续作为主线；
- 原生 JEV、Kev 概率题指标和直接动作 JSON 大语言模型指标按上述协议分别保留；
- 精确数值直接输出暂缓；
- 不实施 Kev 的自回归 JSON 输出头改造。
