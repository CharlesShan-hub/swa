# 电压估计算法模块（深度学习）
#
# 传统 ML / 线性模型已移至 scripts/traditional/
#
# 每个文件一种算法，统一接口：
#   NAME: str          — 算法名称
#   train(...) → model  — 训练
#   predict(model, X) → y_pred — 预测
