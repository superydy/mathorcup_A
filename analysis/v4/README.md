# v4核心验证与冻结

本目录的脚本不覆盖v3.1结果。核心方法与取舍见`docs/解题思路.md`，原始规则和顺序追加记录见`运行前协议.md`，所有新结果位于`results/v4`。

已完成的优化探索已整理至`analysis/exploration_20261026`，含完整OOF预测、内层选择、Q2模块联合置换、Q3协方差探索和环境记录。导入时只复制结果与代码，没有重跑旧模型。`import_exploration.py`是本机一次性导入辅助，完整克隆后无需运行。

运行环境使用已记录的Python 3.9、scikit-learn 1.6.1等。依次运行：

```text
python analysis/v4/q1_blend.py
python analysis/v4/q3_classifier.py
python analysis/v4/q3_classifier_nested.py
python analysis/v4/q3_conditional_ot.py
python analysis/v4/freeze.py
```

所有选择在外层训练集内完成，原外层划分此前已使用，因此结果是内部方法开发证据，不是新的独立确认。尚未完成最终融合模型Q2的100次bootstrap、保形区间、ALE、学习曲线和正文同步，状态明确保存在`results/v4/v4_freeze.json`。

门控L2未超过强基线，不继续将PSO作为冻结前置条件；分类器未达到门槛，只作诊断。低支持度、条件高斯近似及缺失导致不可完整验收的状态均保留。
